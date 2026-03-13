import os
import sys
import argparse
import logging
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration

from model.VideoRestorationModel import VideoRestorationSystem

logger = logging.getLogger(__name__)

def sample_wan_timesteps(n, device, shift=3.0):
    """
    Args:
        n: Batch size
        device: device
        shift: shift factor.
    """
    t = torch.rand((n,), device=device)
    
    if shift != 1.0:
        t_shifted = (shift * t) / (1 + (shift - 1) * t)
    else:
        t_shifted = t
    timesteps = t_shifted * 1000.0
    
    return timesteps.clamp(min=0.001, max=1000.0)

                                                
class WanLatentDataset(Dataset):
    def __init__(self, data_dir):
        self.gt_dir = os.path.join(data_dir, "gt")
        self.cond_dir = os.path.join(data_dir, "cond")
        self.text_embeds_dir = os.path.join(data_dir, "text_embeds")
        
        if os.path.exists(self.gt_dir):
            self.files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith('.pt')])
        else:
            raise FileNotFoundError(f"找不到 GT 目录: {self.gt_dir}")
        self.data_dir = data_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        gt_latent = torch.load(os.path.join(self.gt_dir, filename), map_location="cpu", weights_only=True).squeeze(0).float()
        cond_latent = torch.load(os.path.join(self.cond_dir, filename), map_location="cpu", weights_only=True).squeeze(0).float()
        
        text_embed_path = os.path.join(self.text_embeds_dir, filename)
        if os.path.exists(text_embed_path):
            prompt_embeds = torch.load(text_embed_path, map_location="cpu", weights_only=True).squeeze(0).float()
        else:
            raise FileNotFoundError(f"找不到 Text Embedding: {text_embed_path}. 请先运行 preprocess.py")
            
        return {
            "latents": gt_latent,
            "condition_latents": cond_latent,
            "prompt_embeds": prompt_embeds,
            "filename": filename                                   
        }

def collate_fn(batch):
    latents = torch.stack([item["latents"] for item in batch])
    condition_latents = torch.stack([item["condition_latents"] for item in batch])
    prompt_embeds = torch.stack([item["prompt_embeds"] for item in batch])
    filenames = [item["filename"] for item in batch]                                        
    
    return {
        "latents": latents, 
        "condition_latents": condition_latents, 
        "prompt_embeds": prompt_embeds,
        "filenames": filenames                                       
    }

                            
def main():
    parser = argparse.ArgumentParser(description="Wan 2.2 I2V LoRA Training")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Wan 2.2 Diffusers 模型路径")
    parser.add_argument("--data_dir", type=str, required=True, help="预处理后的 Latents 目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    parser.add_argument("--batch_size", type=int, default=1, help="单卡 Batch Size")
    parser.add_argument("--epochs", type=int, default=10, help="建议 10-30")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="1e-4")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="1e-2")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--mixed_precision", type=str, default="bf16", help="混合精度 (bf16/fp16/no)")
    
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--target_modules", nargs="+", 
                        default=["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"],
                        help="指定需要训练 LoRA 的层名称")

    args = parser.parse_args()
    
    mp = args.mixed_precision
    if mp not in {"bf16", "fp16", "no"}:
        raise ValueError(f"Unsupported mixed_precision: {mp}")

                                      
    logging_dir = os.path.join(args.output_dir, "logs")
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=mp,
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=args.output_dir, 
            logging_dir=logging_dir
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        logger.info(f"🚀 Training on {accelerator.num_processes} GPUs")
        logger.info(f"Configuration: BS={args.batch_size}, LR={args.learning_rate}")
        
        config = {k: (str(v) if isinstance(v, list) else v) for k, v in vars(args).items()}
        accelerator.init_trackers(project_name="wan_lora_training", config=config)

                          
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    model = VideoRestorationSystem(
        pretrained_model_path=args.ckpt_dir,
        dtype=weight_dtype,
        lora_rank=args.lora_rank
    )

    model.setup_lora_training(
        rank=args.lora_rank, 
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules
    )
    
    model.transformer.enable_gradient_checkpointing()
    if model.has_moe and model.transformer_2 is not None:
        model.transformer_2.enable_gradient_checkpointing()

    print("🧹 Cleaning up VAE and Text Encoder to save VRAM...")
    if hasattr(model, "vae"):
        del model.vae
    if hasattr(model, "text_encoder"):
        del model.text_encoder
    torch.cuda.empty_cache()

    model.to(accelerator.device)
    
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    dataset = WanLatentDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1}/{args.epochs} Started")
            
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for batch_idx, batch in enumerate(progress_bar):
            latents = batch["latents"]
            condition = batch["condition_latents"] 
            prompt_embeds = batch["prompt_embeds"]
            filenames = batch["filenames"]                                                      
            
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                
                B = latents.shape[0]
                t = sample_wan_timesteps(B, latents.device, shift=5.0)

                t_cont = t.to(dtype=torch.float32)
                noise = torch.randn_like(latents)
                t_norm = t_cont / 1000.0
                t_expand = t_norm.view(B, 1, 1, 1, 1)
                
                noisy_latents = (1.0 - t_expand) * latents + t_expand * noise
                target = noise - latents

                timesteps_model = t.round().long()
                
                input_dtype = getattr(model, "dtype", noisy_latents.dtype)
                noisy_latents = noisy_latents.to(dtype=input_dtype)
                condition = condition.to(dtype=input_dtype)
                prompt_embeds = prompt_embeds.to(dtype=input_dtype)

                pred_velocity = model(
                    noisy_latents=noisy_latents,
                    timestep=timesteps_model,                 
                    prompt_embeds=prompt_embeds,
                    condition_latents_visual=condition
                )
                
                if torch.isnan(pred_velocity).any():
                    print(f"🚨 [Rank {accelerator.process_index}] NaN Detected! File: {filenames}")
                    continue
                
                pred_velocity_f32 = pred_velocity.to(torch.float32)
                target_f32 = target.to(torch.float32)
                loss = F.mse_loss(pred_velocity_f32, target_f32)
                
                                                                                 
                                                                              
                current_loss = loss.item()
                if current_loss > 0.5:
                                                                         
                                                                                             
                    print(f"🚨 [High Loss] Step: {global_step} | Rank: {accelerator.process_index} | Loss: {current_loss:.4f} | File: {filenames}")

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"🚨 [Rank {accelerator.process_index}] Loss is Inf/NaN! File: {filenames}")
                    continue
        
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                
                optimizer.step()
            
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.set_postfix({"loss": loss.item()})
                
                accelerator.log({"train_loss": loss.item(), "learning_rate": args.learning_rate}, step=global_step)

                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    logger.info(f"Saving LoRA to {save_path}")
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.transformer.save_pretrained(save_path)
                    if unwrapped.has_moe and unwrapped.transformer_2:
                        unwrapped.transformer_2.save_pretrained(os.path.join(save_path, "transformer_2"))

    if accelerator.is_main_process:
        logger.info("Training Finished. Saving final model...")
        final_save_path = os.path.join(args.output_dir, "final_lora")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.transformer.save_pretrained(final_save_path)
        if unwrapped.has_moe and unwrapped.transformer_2:
            unwrapped.transformer_2.save_pretrained(os.path.join(final_save_path, "transformer_2"))
            
    accelerator.end_training()

if __name__ == "__main__":
    main()