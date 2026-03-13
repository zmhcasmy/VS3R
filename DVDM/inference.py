import os
import argparse
import glob
import time
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import numpy as np
from peft import PeftModel
import traceback
from tqdm import tqdm
from model.VideoRestorationModel import VideoRestorationSystem

def load_and_preprocess_video(video_path, target_h=480, target_w=832, max_frames=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path, 'ffmpeg')
    
    try:
        fps = reader.get_meta_data()['fps']
    except Exception:
        fps = 24.0
        print(f"⚠️ Warning: Could not read FPS for {video_path}, defaulting to 24.")

    frames = []
    for i, im in enumerate(reader):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(im)
    
    if not frames:
        raise RuntimeError("No frames loaded.")

    video = torch.tensor(np.array(frames)).permute(3, 0, 1, 2).float()
    
    curr_h, curr_w = video.shape[2], video.shape[3]
    scale = max(target_h / curr_h, target_w / curr_w)
    new_h, new_w = int(curr_h * scale), int(curr_w * scale)
    
    if new_h != curr_h or new_w != curr_w:
        temp = video.permute(1, 0, 2, 3) 
        temp = F.interpolate(temp, size=(new_h, new_w), mode='bilinear', align_corners=False)
        video = temp.permute(1, 0, 2, 3)

    start_y = (new_h - target_h) // 2
    start_x = (new_w - target_w) // 2
    video = video[:, :, start_y:start_y+target_h, start_x:start_x+target_w]
    
    video = (video / 255.0) * 2.0 - 1.0
    return video, fps

def save_video(tensor, path, fps=24):
    if tensor.dim() == 5: tensor = tensor.squeeze(0)
    tensor = (tensor.detach().float() * 0.5 + 0.5).clamp(0, 1)
    frames = (tensor.permute(1, 2, 3, 0).cpu().numpy() * 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    writer = imageio.get_writer(path, format="FFMPEG", fps=fps, codec="libx264", pixelformat="yuv420p")
    try:
        for f in frames: writer.append_data(f)
    finally:
        writer.close()

def format_gb(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f} GB"

def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.2f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.2f}s"

def get_blending_weights(window_size, overlap):
    weights = torch.ones(window_size)
    if overlap > 0:
        weights[:overlap] = torch.linspace(0.0, 1.0, overlap)
        weights[-overlap:] = torch.linspace(1.0, 0.0, overlap)
    return weights

                                                
def process_long_video_sliding_window(
    model,
    video_tensor,
    prompt,
    denoise,
    window_size=81,
    overlap=21,
    sampling_steps=40,
    debug_folder=None,
    fps=24,
):
    C, TotalFrames, H, W = video_tensor.shape
    device = model.device
    
    final_output = torch.zeros((C, TotalFrames, H, W), dtype=torch.float32)
    weight_accumulator = torch.zeros((1, TotalFrames, 1, 1), dtype=torch.float32)
    
    stride = window_size - overlap
    total_chunks = (TotalFrames + stride - 1) // stride

    if debug_folder:
        os.makedirs(debug_folder, exist_ok=True)

    for start_idx in tqdm(range(0, TotalFrames, stride), total=total_chunks, desc="Restoring", unit="chunk", dynamic_ncols=True):
        end_idx = start_idx + window_size
        
        if end_idx > TotalFrames:
            if start_idx == 0: 
                end_idx = TotalFrames
            else:
                end_idx = TotalFrames
                start_idx = max(0, TotalFrames - window_size)
        
        chunk = video_tensor[:, start_idx:end_idx, :, :].unsqueeze(0)
        current_len = chunk.shape[2] 
        
        with torch.inference_mode():
            restored_chunk = model.restore(
                video_input=chunk, 
                prompt=prompt,
                denoise_strength=denoise,
                frames=current_len,
                sampling_steps=sampling_steps
            )
        
        restored_chunk = restored_chunk.cpu().squeeze(0)

        if debug_folder:
            chunk_name = f"chunk_{start_idx:04d}_{end_idx:04d}.mp4"
            chunk_path = os.path.join(debug_folder, chunk_name)
            save_video(restored_chunk, chunk_path, fps=fps)

        weights = get_blending_weights(current_len, overlap)
        
        if start_idx == 0:
            weights[:overlap] = 1.0
        if end_idx == TotalFrames:
            weights[-overlap:] = 1.0
            
        weights = weights.view(1, -1, 1, 1) 
        
        final_output[:, start_idx:end_idx, :, :] += restored_chunk * weights
        weight_accumulator[:, start_idx:end_idx, :, :] += weights

        if end_idx == TotalFrames:
            break
            
    final_output = final_output / (weight_accumulator + 1e-6)
    return final_output

def run_vae_sanity_check(model, video_tensor):
    print("\n🧪 ====== VAE Sanity Check Mode ======")
    device = model.device
    dtype = model.dtype
    
    check_len = min(video_tensor.shape[1], 81)
    chunk = video_tensor[:, :check_len, :, :]
    
    with torch.no_grad():
        latents = model.encode_video(chunk)
        recon_video = model.decode_latents(latents)
    return recon_video

def process_single_file(video_path, output_root, args):
    video_name = os.path.basename(video_path)
    name_no_ext = os.path.splitext(video_name)[0]
    out_video_path = os.path.join(output_root, f"{name_no_ext}.mp4")
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

    if os.path.exists(out_video_path):
        print(f"⏭️  Skipping existing file: {out_video_path}")
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device) 
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model = VideoRestorationSystem(args.ckpt_dir, dtype=dtype, offload_model=args.offload_model)
    
    if args.lora_path and not args.check_vae:
        main_lora = os.path.join(args.lora_path, "transformer")
        if os.path.exists(main_lora):
            model.transformer = PeftModel.from_pretrained(model.transformer, main_lora)
            model.transformer = model.transformer.merge_and_unload()
        if model.has_moe:
            moe_lora = os.path.join(args.lora_path, "transformer_2")
            if os.path.exists(moe_lora):
                model.transformer_2 = PeftModel.from_pretrained(model.transformer_2, moe_lora)
                model.transformer_2 = model.transformer_2.merge_and_unload()

    if not args.offload_model:
        model.to(device)
    model.eval()

    try:
        max_frames = None if args.frames <= 0 else args.frames
        video_tensor, fps = load_and_preprocess_video(
            video_path, 
            target_h=args.height, 
            target_w=args.width, 
            max_frames=max_frames
        )

        if args.check_vae:
            debug_output = os.path.join(output_root, f"{video_name}_vae_debug.mp4")
            start_time = time.time()
            recon_video = run_vae_sanity_check(model, video_tensor)
            elapsed = time.time() - start_time
            processed_frames = min(video_tensor.shape[1], 81)
            save_video(recon_video, debug_output, fps=fps)
        else:
            total_frames = video_tensor.shape[1]
            device = model.device
            
            if total_frames <= args.window_size:
                start_time = time.time()
                with torch.inference_mode():
                    restored = model.restore(
                        video_input=video_tensor,
                        prompt=args.prompt,
                        denoise_strength=args.denoise,
                        frames=total_frames,
                        sampling_steps=40
                    ).cpu()
                elapsed = time.time() - start_time
            else:
                debug_dir = None
                
                start_time = time.time()
                restored = process_long_video_sliding_window(
                    model=model,
                    video_tensor=video_tensor, 
                    prompt=args.prompt,
                    denoise=args.denoise,
                    window_size=args.window_size,
                    overlap=args.overlap,
                    debug_folder=debug_dir,
                    fps=fps
                )
                elapsed = time.time() - start_time
                
                trim_count = args.trim_last
                if trim_count == -1:
                    trim_count = args.overlap 
                
                if trim_count > 0 and restored.shape[1] > trim_count:
                    restored = restored[:, :-trim_count, :, :]

            processed_frames = restored.shape[1]
            save_video(restored, out_video_path, fps=fps)

        if processed_frames > 0:
            per_frame = elapsed / processed_frames
            print(f"Total Time: {format_seconds(elapsed)}")
            print(f"Per-Frame Time: {format_seconds(per_frame)}")
            
    except Exception as e:
        print(f"Error processing {video_name}: {e}")
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            print(f"GPU Peak Allocated: {format_gb(peak_alloc)}")
            print(f"GPU Peak Reserved: {format_gb(peak_reserved)}")
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save output video")
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--lora_path", type=str, default="../ckpts/lora")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=0) 
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="high quality video, clean, stable, 3d render, point cloud restoration")
    parser.add_argument("--check_vae", action="store_true")
    parser.add_argument("--offload_model", action="store_true")
    
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--trim_last", type=int, default=-1)

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    
    if not os.path.exists(args.video_path):
        print(f"Error: Input video not found: {args.video_path}")
        return

    process_single_file(args.video_path, args.output_folder, args)

if __name__ == "__main__":
    main()
