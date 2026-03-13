import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from fixpipeline.VideoRestorationModel_diffuser import VideoRestorationSystem

def process_frame_chunk(frames_list, width=832, height=480):
    video = torch.tensor(np.array(frames_list)).float()
    
    video = video.permute(0, 3, 1, 2)
    
    video = F.interpolate(video, size=(height, width), mode='bilinear', align_corners=False)
    
    video = (video / 255.0) * 2.0 - 1.0
    
    video = video.permute(1, 0, 2, 3).unsqueeze(0)
    return video

def get_video_chunks(path, chunk_size=81, width=832, height=480):
    try:
        reader = imageio.get_reader(path, 'ffmpeg')
        all_frames = []
        for im in reader:
            all_frames.append(im)
        
        total_frames = len(all_frames)
        
        if total_frames == 0:
            print(f"Error: Empty video {path}")
            return []

        if total_frames < chunk_size:
            frames_to_process = all_frames.copy()
            while len(frames_to_process) < chunk_size:
                frames_to_process.append(frames_to_process[-1])
            tensor = process_frame_chunk(frames_to_process, width, height)
            return [tensor]

        num_chunks = total_frames // chunk_size
        tensor_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk_frames = all_frames[start_idx:end_idx]
            tensor = process_frame_chunk(chunk_frames, width, height)
            tensor_chunks.append(tensor)
            
        return tensor_chunks
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/zmh/gitclone/Wan2.2/Wan2.2-I2V-A14B-Diffusers", help="Wan2.2 Diffusers 模型路径")
    parser.add_argument("--gt_dir", type=str, default="/home/zmh/program/video_stablization/dataset/train_data/stable_video", help="GT 视频路径")
    parser.add_argument("--cond_dir", type=str, default="/home/zmh/program/video_stablization/dataset/train_data/stable_video_reproject_with_artifact", help="Condition 视频路径")
    parser.add_argument("--output_dir", type=str, default="/home/zmh/program/video_stablization/dataset/train_data/latents", help="输出目录")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--prompt", type=str, default=None)
    
    args = parser.parse_args()
    
    device = "cuda"
    dtype = torch.bfloat16
    
    if not os.path.exists(args.gt_dir):
        raise FileNotFoundError(f"GT 目录不存在: {args.gt_dir}")
    if not os.path.exists(args.cond_dir):
        raise FileNotFoundError(f"Condition 目录不存在: {args.cond_dir}")
    
    print(f"Initializing System from {args.model_path}...")
    system = VideoRestorationSystem(args.model_path, dtype=dtype).to(device)
    system.eval()
    
    torch.cuda.empty_cache()
    
    os.makedirs(os.path.join(args.output_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "cond"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "text_embeds"), exist_ok=True)
    
    files = [f for f in os.listdir(args.gt_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"Found {len(files)} videos in GT dir. Starting preprocessing...")
    
    valid_exts = ['.mp4', '.avi', '.mov', '.mkv']

    with torch.no_grad():
        for f in tqdm(files):
            basename = os.path.splitext(f)[0]
            
            cond_path = None
            found_ext = ""
            for ext in valid_exts:
                test_path = os.path.join(args.cond_dir, basename + ext)
                if os.path.exists(test_path):
                    cond_path = test_path
                    found_ext = ext
                    break
            
            if cond_path is None:
                tqdm.write(f"[Skip] No condition file for: {basename} (checked: {valid_exts})")
                continue

            gt_path = os.path.join(args.gt_dir, f)
            gt_chunks = get_video_chunks(gt_path, args.frames, args.width, args.height)
            
            if not gt_chunks:
                continue
            
            cond_chunks = get_video_chunks(cond_path, args.frames, args.width, args.height)
            
            if not cond_chunks:
                tqdm.write(f"[Skip] Condition video empty or failed: {basename}{found_ext}")
                continue

            num_chunks_gt = len(gt_chunks)
            num_chunks_cond = len(cond_chunks)
            
            final_num_chunks = min(num_chunks_gt, num_chunks_cond)
            
            if num_chunks_gt != num_chunks_cond:
                tqdm.write(
                    f"⚠️  Alignment: {basename} | GT: {num_chunks_gt} chunks, Cond: {num_chunks_cond} chunks. "
                    f"-> Using first {final_num_chunks} chunks."
                )

            txt_path = os.path.join(args.gt_dir, f"{basename}.txt")
            if args.prompt is not None:
                prompt = args.prompt
            elif os.path.exists(txt_path):
                with open(txt_path, 'r') as tf:
                    prompt = tf.read().strip()
            else:
                prompt = "high quality video, clean, stable, 3d render, point cloud restoration"
            
            prompt_embeds = system.encode_text(prompt).cpu()

            for i in range(final_num_chunks):
                save_name = f"{basename}_{i+1}"
                
                gt_vid = gt_chunks[i].to(device, dtype=dtype)
                gt_latent = system.encode_video(gt_vid).cpu()
                torch.save(gt_latent, os.path.join(args.output_dir, "gt", f"{save_name}.pt"))
                
                cond_vid = cond_chunks[i].to(device, dtype=dtype)
                cond_latent = system.encode_video(cond_vid).cpu()
                torch.save(cond_latent, os.path.join(args.output_dir, "cond", f"{save_name}.pt"))
                
                torch.save(prompt_embeds, os.path.join(args.output_dir, "text_embeds", f"{save_name}.pt"))

    print("✅ Preprocessing Complete!")

if __name__ == "__main__":
    main()
