                                                                        
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional, Dict, Any, List

                             
try:
    from diffusers import (
        WanTransformer3DModel, 
        AutoencoderKLWan, 
        FlowMatchEulerDiscreteScheduler,
        WanImageToVideoPipeline
    )
    from transformers import UMT5EncoderModel, T5TokenizerFast
except ImportError:
    logging.warning("Wan2.2 specific diffusers classes not found. Ensure you have the correct branch installed.")
    pass

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    logging.warning("peft library not found. LoRA training will not be available.")

class VideoRestorationSystem(nn.Module):
    """
    A system for Video Restoration using Wan2.2 components.
    修正了 VAE Latent 的归一化逻辑 (Channel-wise Mean/Std)，适配 Wan2.2 架构。
    适配 Accelerate 分布式训练 (移除硬编码 device)。
    """
    def __init__(
        self,
        pretrained_model_path: str,
                                            
        dtype: torch.dtype = torch.bfloat16,
        lora_rank: int = 32,
        lora_alpha: int = 32,
        offload_model: bool = False,
        **kwargs
    ):
        super().__init__()
                                   
        self.dtype = dtype
        self.pretrained_model_path = pretrained_model_path
        self.offload_model = offload_model
        
        print(f"Initializing WanRestorationSystem from {pretrained_model_path}...")
        self._init_components()
        self.freeze_base_models()

    @property
    def device(self):
        """
        动态获取设备。
        优先取 transformer，如果 transformer 被卸载（如预处理时），
        则尝试取 text_encoder 或 vae 的设备。
        """
        if self.offload_model and torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(self, 'transformer') and self.transformer is not None:
            return self.transformer.device
    
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            return self.text_encoder.device

        if hasattr(self, 'vae') and self.vae is not None:
            return self.vae.device
            
        return torch.device("cuda")
        
    def _init_components(self):
        """Initialize all sub-models from the diffusers checkpoint."""
                                     
        self.tokenizer = T5TokenizerFast.from_pretrained(
            self.pretrained_model_path, subfolder="tokenizer"
        )
                                            
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.pretrained_model_path, subfolder="text_encoder", torch_dtype=self.dtype
        )
        
                
        try:
                                      
            self.vae = AutoencoderKLWan.from_pretrained(
                self.pretrained_model_path, subfolder="vae", torch_dtype=self.dtype
            )
        except NameError:
            from diffusers import AutoencoderKL
                                      
            self.vae = AutoencoderKL.from_pretrained(
                self.pretrained_model_path, subfolder="vae", torch_dtype=self.dtype
            )

                              
                                  
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.pretrained_model_path, subfolder="transformer", torch_dtype=self.dtype
        )
        
        try:
                                      
            self.transformer_2 = WanTransformer3DModel.from_pretrained(
                self.pretrained_model_path, subfolder="transformer_2", torch_dtype=self.dtype
            )
            self.has_moe = True
            print("Loaded transformer_2 (Low Noise Expert).")
        except Exception:
            self.transformer_2 = None
            self.has_moe = False
            print("Only one transformer found.")

                      
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_path, subfolder="scheduler"
        )

    def freeze_base_models(self):
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if self.has_moe:
            self.transformer_2.requires_grad_(False)

    def setup_lora_training(self, rank=32, lora_alpha=32, target_modules=None):
        if target_modules is None:
            target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]

        print(f"Setting up LoRA training with rank={rank}...")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.print_trainable_parameters()
        
        if self.has_moe:
            self.transformer_2 = get_peft_model(self.transformer_2, lora_config)
            self.transformer_2.print_trainable_parameters()

        self.transformer.train()
        if self.has_moe:
            self.transformer_2.train()

    def load_lora_weights(self, lora_path: str, adapter_name="default"):
        print(f"Loading LoRA weights from {lora_path}...")
        try:
            self.transformer.load_lora_weights(lora_path, adapter_name=adapter_name)
            if self.has_moe:
                try:
                    self.transformer_2.load_lora_weights(lora_path, adapter_name=adapter_name)
                except Exception as e:
                    print(f"Warning: Could not load LoRA for transformer_2: {e}")
        except AttributeError:
             pass

    @torch.no_grad()
    def encode_text(self, prompt: str, max_length: int = 226):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_device = self.text_encoder.device
        target_device = self.device
        if self.offload_model and text_device.type == "cpu" and target_device.type == "cuda":
            self.text_encoder.to(target_device)
            text_device = self.text_encoder.device
        text_input_ids = text_inputs.input_ids.to(text_device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        if prompt_embeds.device != target_device:
            prompt_embeds = prompt_embeds.to(target_device)
        if self.offload_model and self.text_encoder.device.type == "cuda":
            self.text_encoder.to("cpu")
        return prompt_embeds

                                                
    def _get_latents_mean_std(self, device, dtype):
        """
        Helper to get mean and std tensors reshaped for broadcasting.
        Wan2.2 VAE 使用 Channel-wise 归一化。
        """
        mean_list = getattr(self.vae.config, "latents_mean", [0.0] * 16)
        std_list = getattr(self.vae.config, "latents_std", [1.0] * 16)
        
                              
        mean = torch.tensor(mean_list, device=device, dtype=dtype)
        std = torch.tensor(std_list, device=device, dtype=dtype)
        
        mean = mean.view(1, -1, 1, 1, 1)
        std = std.view(1, -1, 1, 1, 1)
        
        return mean, std

    @torch.no_grad()
    def encode_video(self, video_tensor: torch.Tensor):
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)
        
        vae_device = self.vae.device
        target_device = self.device
        if self.offload_model and vae_device.type == "cpu" and target_device.type == "cuda":
            self.vae.to(target_device)
            vae_device = self.vae.device
        video_tensor = video_tensor.to(vae_device, dtype=self.dtype)
        dist = self.vae.encode(video_tensor).latent_dist
        latents = dist.sample()
        
        mean, std = self._get_latents_mean_std(latents.device, latents.dtype)
        latents = (latents - mean) / std
        
        if latents.device != target_device:
            latents = latents.to(target_device)
        if self.offload_model and self.vae.device.type == "cuda":
            self.vae.to("cpu")
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        target_device = self.device
        vae_device = self.vae.device
        if self.offload_model and vae_device.type == "cpu" and target_device.type == "cuda":
            self.vae.to(target_device)
            vae_device = self.vae.device
        if latents.device != vae_device:
            latents = latents.to(vae_device)
        mean, std = self._get_latents_mean_std(latents.device, latents.dtype)
        latents = latents * std + mean
        
        video = self.vae.decode(latents).sample
        if self.offload_model and self.vae.device.type == "cuda":
            self.vae.to("cpu")
        return video

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        condition_latents_visual: torch.Tensor,
    ):
        """
        Forward pass for prediction using Dense Conditioning.
        """
                                 
        current_device = noisy_latents.device
        current_dtype = noisy_latents.dtype

        B = noisy_latents.shape[0]
        _, _, F_lat, H_lat, W_lat = condition_latents_visual.shape
        
                                                
        mask = torch.ones((B, 4, F_lat, H_lat, W_lat), device=current_device, dtype=current_dtype)
        
                      
        condition_latents = torch.cat([mask, condition_latents_visual], dim=1)
        
              
        latent_model_input = torch.cat([noisy_latents, condition_latents], dim=1)

                                                             
        
                   
        if self.has_moe:
                                                     
            output = torch.zeros_like(noisy_latents)
            
            t_threshold = 900.0
            high_noise_mask = timestep >= t_threshold
            low_noise_mask = timestep < t_threshold
            if high_noise_mask.any():
                                       
                res_high = self.transformer(
                    hidden_states=latent_model_input[high_noise_mask],
                    timestep=timestep[high_noise_mask],
                    encoder_hidden_states=prompt_embeds[high_noise_mask],
                    return_dict=False
                )[0]
                
                                              
                output[high_noise_mask] = res_high.to(output.dtype)
                
                                         
            if low_noise_mask.any():
                                       
                res_low = self.transformer_2(
                    hidden_states=latent_model_input[low_noise_mask],
                    timestep=timestep[low_noise_mask],
                    encoder_hidden_states=prompt_embeds[low_noise_mask],
                    return_dict=False
                )[0]
                
                                              
                output[low_noise_mask] = res_low.to(output.dtype)
            
            return output
        else:
                                              
            return self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]

    def _prepare_model_for_timestep(self, t):
        if not self.has_moe:
            if self.offload_model and self.transformer.device.type == "cpu":
                self.transformer.to(self.device)
            return self.transformer

        if t.item() < 900:
            required_model = self.transformer_2
            offload_model = self.transformer
        else:
            required_model = self.transformer
            offload_model = self.transformer_2

        if self.offload_model:
            if offload_model is not None and next(offload_model.parameters()).device.type == "cuda":
                offload_model.to("cpu")
            if required_model is not None and next(required_model.parameters()).device.type == "cpu":
                required_model.to(self.device)
        return required_model

    def restore(
            self,
            video_input: torch.Tensor,
            prompt: str,
            denoise_strength: float = 1.0,                      
            frames: int = 81,
            sampling_steps: int = 40,
            guidance_scale: float = 5.0,
            seed: int = 42
        ):
            """
            Main restoration loop with proper Denoising Strength support.
            """
            device = self.device 
            dtype = self.dtype
            
            generator = torch.Generator(device=device).manual_seed(seed)
            batch_size = 1
            
                                             
                                                     
                                                     
            visual_latents = self.encode_video(video_input.to(device, dtype=dtype))
            
                                  
            B, C, F_lat, H_lat, W_lat = visual_latents.shape
            mask = torch.ones((B, 4, F_lat, H_lat, W_lat), device=device, dtype=dtype)

                                                                                         
        
                                        
                                    
                                       
            
            condition_latents = torch.cat([ mask,visual_latents], dim=1) 
            
                                           
            self.scheduler.set_timesteps(sampling_steps, device=device)
            timesteps = self.scheduler.timesteps
            
                                           
            num_inference_steps = len(timesteps)
            
                                            
            denoise_strength = min(max(denoise_strength, 0.0), 1.0)
            
                        
                                                    
                                                          
            init_timestep_idx = min(int(num_inference_steps * (1 - denoise_strength)), num_inference_steps - 1)
            
                         
            t_start = timesteps[init_timestep_idx]
            timesteps = timesteps[init_timestep_idx:]
            
            
                                                  
                                                                     
                     
            noise = torch.randn(
                (batch_size, 16, F_lat, H_lat, W_lat), 
                device=device, 
                dtype=dtype,
                generator=generator
            )
            
            if denoise_strength >= 1.0:
                                       
                latents = noise
            else:
                                                            
                                                               
                sigma = t_start.item() / 1000.0
                latents = (1 - sigma) * visual_latents + sigma * noise

                            
            prompt_embeds = self.encode_text(prompt)
            
            autocast_ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), autocast_ctx:
                for t in timesteps:
                    model = self._prepare_model_for_timestep(t)
                    
                    latent_model_input = torch.cat([latents, condition_latents], dim=1)
                    
                    t_batch = t.expand(latent_model_input.shape[0])

                    noise_pred = model(
                        hidden_states=latent_model_input,
                        timestep=t_batch,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False
                    )[0]
                    
                    if self.offload_model and device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            restored_video = self.decode_latents(latents)
            if self.offload_model and device.type == "cuda":
                if self.transformer.device.type == "cuda":
                    self.transformer.to("cpu")
                if self.has_moe and self.transformer_2 is not None and self.transformer_2.device.type == "cuda":
                    self.transformer_2.to("cpu")
                torch.cuda.empty_cache()
            return restored_video
