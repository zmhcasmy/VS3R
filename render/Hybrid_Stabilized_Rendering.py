import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
import shutil
import tempfile
import argparse
from contextlib import contextmanager
import torch
import numpy as np
import cv2
import imageio.v2 as imageio
import scipy.signal
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from einops import rearrange
from tqdm import tqdm
torch.backends.cudnn.enabled = False

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.masks.dynamic_mask import (adaptive_multiotsu_variance,
                                             cluster_attention_maps,
                                             extract_dyn_map)
import vggt4d.masks.dynamic_mask as dynamic_mask_module
import vggt4d.masks.refine_dyn_mask as refine_dyn_mask_module
from vggt4d.utils.model_utils import inference, organize_qk_dict
from vggt4d.utils.store import (save_depth, save_depth_conf,
                                save_dynamic_masks, save_intrinsic_txt,
                                save_rgb, save_tum_poses)

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from pytorch3d.renderer import PointsRasterizationSettings, PointsRenderer, PointsRasterizer, AlphaCompositor
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras
try:
    from camera.fisheye_camera import FishEyeCameras, FISHEYE_RADIAL, FISHEYE_TANGENTIAL, FISHEYE_THIN_PRISM
except ImportError:
    try:
        from render.camera.fisheye_camera import FishEyeCameras, FISHEYE_RADIAL, FISHEYE_TANGENTIAL, FISHEYE_THIN_PRISM
    except ImportError:
        FishEyeCameras = None
        FISHEYE_RADIAL = None
        FISHEYE_TANGENTIAL = None
        FISHEYE_THIN_PRISM = None
        print("[Warning] Could not import FishEyeCameras from local module.")
try:
    from camera.ucm_camera import UnifiedOmnidirectionalCameras, UCM_XI
except ImportError:
    try:
        from render.camera.ucm_camera import UnifiedOmnidirectionalCameras, UCM_XI
    except ImportError:
        UnifiedOmnidirectionalCameras = None
        UCM_XI = None
        print("[Warning] Could not import UnifiedOmnidirectionalCameras from local module.")
try:
    from camera.double_sphere_camera import DoubleSphereCameras, DSM_XI, DSM_ALPHA
except ImportError:
    try:
        from render.camera.double_sphere_camera import DoubleSphereCameras, DSM_XI, DSM_ALPHA
    except ImportError:
        DoubleSphereCameras = None
        DSM_XI = None
        DSM_ALPHA = None
        print("[Warning] Could not import DoubleSphereCameras from local module.")


def select_device():
    if torch.cuda.is_available():
        torch.empty(1, device="cuda:0")
        dev = "cuda:0"
    else:
        dev = "cpu"
    print(f"device={dev}")
    return dev

@contextmanager
def suppress_output():
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

@contextmanager
def suppress_internal_tqdm():
    dm_tqdm = dynamic_mask_module.tqdm
    rdm_tqdm = refine_dyn_mask_module.tqdm
    dynamic_mask_module.tqdm = lambda x, *a, **k: x
    refine_dyn_mask_module.tqdm = lambda x, *a, **k: x
    try:
        yield
    finally:
        dynamic_mask_module.tqdm = dm_tqdm
        refine_dyn_mask_module.tqdm = rdm_tqdm

class InputPadder:
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def load_raft_model(device):
    print("正在加载 RAFT 光流模型...")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device)
    model.eval()
    return model, weights.transforms()


def compute_rigid_flow(depth, pose1, pose2, intrinsic, H, W):
    device = depth.device
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixel_coords = torch.stack([x, y], dim=-1).float()
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    depth = torch.clamp(depth, min=1e-5)
    X = (pixel_coords[..., 0] - cx) * depth / fx
    Y = (pixel_coords[..., 1] - cy) * depth / fy
    Z = depth
    points_cam1 = torch.stack([X, Y, Z], dim=-1)
    ones = torch.ones_like(Z).unsqueeze(-1)
    points_cam1_hom = torch.cat([points_cam1, ones], dim=-1)
    w2c_2 = torch.linalg.inv(pose2)
    relative_pose = torch.matmul(w2c_2, pose1)
    points_cam2_hom = torch.matmul(points_cam1_hom, relative_pose.T)
    points_cam2 = points_cam2_hom[..., :3]
    Z2 = points_cam2[..., 2]
    Z2_safe = torch.clamp(Z2, min=1e-5)
    u2 = (points_cam2[..., 0] * fx / Z2_safe) + cx
    v2 = (points_cam2[..., 1] * fy / Z2_safe) + cy
    pixel_coords_2 = torch.stack([u2, v2], dim=-1)
    flow_rigid = pixel_coords_2 - pixel_coords
    return flow_rigid

def normalize_depth_shape(depth, H, W):
    if depth.shape == (H, W):
        return depth
    if depth.shape == (W, H):
        return depth.T
    if depth.ndim == 3 and depth.shape[0] == 1 and depth.shape[1:] == (H, W):
        return depth.squeeze(0)
    if depth.ndim == 3 and depth.shape[2] == 1 and depth.shape[:2] == (H, W):
        return depth.squeeze(2)
    print(f"Warning: Depth shape {depth.shape} mismatch Image {(H, W)}, resizing...")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth_input = depth.permute(2, 0, 1).unsqueeze(0)
    elif depth.ndim == 3:
        depth_input = depth.unsqueeze(0)
    else:
        depth_input = depth.unsqueeze(0).unsqueeze(0)
    return F.interpolate(depth_input, size=(H, W), mode='nearest').squeeze()


def compute_dynamic_masks(images_np, depths_np, intrinsics_np, extrinsics_np, raft_model, raft_transform, device, threshold=5.0):
    print(f"正在计算动态掩码 (阈值={threshold} px)...")
    N, H, W, _ = images_np.shape
    masks = []
    c2w_list = []
    for i in range(N):
        ex = extrinsics_np[i]
        if ex.shape == (4, 4):
            w2c = ex
        elif ex.shape == (3, 4):
            ones = np.array([[0, 0, 0, 1]])
            w2c = np.concatenate([ex, ones], axis=0)
        else:
            raise ValueError(f"Invalid extrinsic shape: {ex.shape}")
        c2w = np.linalg.inv(w2c)
        c2w_list.append(torch.from_numpy(c2w).float().to(device))
    intrinsics_t = torch.from_numpy(intrinsics_np).float().to(device)
    for i in range(N):
        if i == N - 1:
            masks.append(np.zeros((H, W), dtype=bool))
            continue
        img1 = torch.from_numpy(images_np[i]).permute(2,0,1).to(device)
        img2 = torch.from_numpy(images_np[i+1]).permute(2,0,1).to(device)
        depth1 = torch.from_numpy(depths_np[i]).float().to(device)
        img1_batch = img1.unsqueeze(0)
        img2_batch = img2.unsqueeze(0)
        img1_batch, img2_batch = raft_transform(img1_batch, img2_batch)
        with torch.no_grad():
            padder = InputPadder(img1_batch.shape)
            img1_batch, img2_batch = padder.pad(img1_batch, img2_batch)
            list_of_flows = raft_model(img1_batch, img2_batch)
            flow_observed = list_of_flows[-1]
            flow_observed = padder.unpad(flow_observed)
            flow_observed = flow_observed[0].permute(1, 2, 0)
            if flow_observed.shape[:2] != (H, W):
                flow_observed = F.interpolate(flow_observed.permute(2,0,1).unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False)
                flow_observed = flow_observed.squeeze(0).permute(1,2,0)
        depth1 = normalize_depth_shape(depth1, H, W)
        flow_rigid = compute_rigid_flow(depth1, c2w_list[i], c2w_list[i+1], intrinsics_t[i], H, W)
        diff = torch.norm(flow_observed - flow_rigid, dim=-1)
        mask_t = diff > threshold
        masks.append(mask_t.cpu().numpy())
    return np.stack(masks, axis=0)

                                                                           
                         
                                                                           

def extract_video_frames(video_path, frame_stride, frames_dir, max_frames=486):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(frames_dir, exist_ok=True)
    paths = []
    w = h = None

    frame_idx = 0
    save_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % max(frame_stride, 1) == 0:
            if w is None:
                h, w = frame.shape[:2]

            p = os.path.join(frames_dir, f"{save_idx:06d}.png")
            cv2.imwrite(p, frame)
            paths.append(p)
            save_idx += 1

            if max_frames is not None and save_idx >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"[提取] 路径={os.path.basename(video_path)}, 帧数={len(paths)}, 尺寸={(h, w)}, fps={fps}")
    return paths, (h, w), fps

def robust_align_chunks(prev_depths, prev_c2ws, curr_depths, curr_c2ws, prev_conf=None, curr_conf=None):
    """Chunk alignment logic (Robust)"""
    valid_mask = (prev_depths > 1e-4) & (curr_depths > 1e-4)
    
    if prev_conf is not None and curr_conf is not None:
                                                             
        conf_mask = (prev_conf > 0.5) & (curr_conf > 0.5)
        valid_mask = valid_mask & conf_mask

    if np.sum(valid_mask) < 50:
        print("  [Warning] Not enough valid depth points for alignment. Defaulting to identity.")
        return 1.0, np.eye(4)
        
    scale_ratios = prev_depths[valid_mask] / curr_depths[valid_mask]
    
                                       
    q1 = np.percentile(scale_ratios, 25)
    q3 = np.percentile(scale_ratios, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    valid_ratios = scale_ratios[(scale_ratios >= lower) & (scale_ratios <= upper)]
    
    if len(valid_ratios) > 10:
        scale = np.median(valid_ratios)
    else:
        scale = np.median(scale_ratios)
    
                                         
    scale = np.clip(scale, 0.8, 1.25)
    
    c2w_curr_anchor = curr_c2ws[0].copy()
    c2w_curr_anchor[:3, 3] *= scale
    
    T_align = prev_c2ws[0] @ np.linalg.inv(c2w_curr_anchor)
    return scale, T_align

def aggregate_point_cloud(points, colors, conf, extrinsic, intrinsic, dynamic_masks=None, conf_thres=0.0):
    """
    Build a global point cloud and attach dynamic/static flags
    dynamic_masks: (S, H, W) boolean array
    Returns: pts_f, rgb, fid, dyn_flag
    """
                                                                             
    if points.ndim == 3:
        points = points[..., None]
        
    pts = unproject_depth_map_to_point_map(points, extrinsic, intrinsic)
    
                                             
    mask = conf >= conf_thres if conf is not None else np.ones_like(pts[..., 0], dtype=bool)
    
    pts_f = pts[mask]
    rgb = (colors.transpose(0, 2, 3, 1)[mask] * 255).astype(np.uint8)
    
    S, H, W = pts.shape[:3]
    frame_ids = np.repeat(np.arange(S), H * W).reshape(S, H, W)
    fid = frame_ids[mask]
    
                          
    if dynamic_masks is not None:
        if dynamic_masks.shape != (S, H, W):
            print(f"[Warning] Mask shape mismatch {dynamic_masks.shape} vs {(S,H,W)}")
            dyn_flat = np.zeros_like(fid, dtype=bool)
        else:
            dyn_flat = dynamic_masks[mask]
    else:
        dyn_flat = np.zeros_like(fid, dtype=bool)
        
    return pts_f, rgb, fid, dyn_flat

def smooth_camera_trajectory_gaussian(poses_mat, smooth_window=59, stability=10): 
    """
    Gaussian Smoothing Strategy adapted for (N, 4, 4) input.
    """
    
    N = poses_mat.shape[0]
    
                               
    window = scipy.signal.windows.gaussian(smooth_window, std=stability)
    window /= window.sum()
    
                           
    t = poses_mat[:, :3, 3]
                                                                                 
    ones = np.ones(N)
    norm_factor = scipy.signal.convolve(ones, window, mode='same')
    
    t_smooth = np.zeros_like(t)
    for i in range(3):
        t_smooth[:, i] = scipy.signal.convolve(t[:, i], window, mode='same') / norm_factor
        
                                         
    R_curr = poses_mat[:, :3, :3]
    quats = Rotation.from_matrix(R_curr).as_quat()
    
                        
    for i in range(1, N):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i]
            
    quats_smooth = np.zeros_like(quats)
    for i in range(4):
        quats_smooth[:, i] = scipy.signal.convolve(quats[:, i], window, mode='same') / norm_factor
        
                           
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)
    
    R_smooth = Rotation.from_quat(quats_smooth).as_matrix()
    
                 
    smooth_poses = np.eye(4)[None].repeat(N, axis=0)
    smooth_poses[:, :3, :3] = R_smooth
    smooth_poses[:, :3, 3] = t_smooth
    
    return smooth_poses

def _expand_param(param, S, device, dim):
    t = torch.tensor(param, device=device, dtype=torch.float32)
    if t.ndim == 0:
        return t.view(1, 1).repeat(S, 1)
    if t.ndim == 1:
        if t.numel() == dim:
            return t.view(1, dim).repeat(S, 1)
        if t.numel() == S and dim == 1:
            return t.view(S, 1)
        raise ValueError(f"Invalid param length: {t.numel()} expected {dim} or {S}")
    if t.ndim == 2:
        if t.shape[0] == 1 and t.shape[1] == dim:
            return t.repeat(S, 1)
        if t.shape[0] == S and t.shape[1] == dim:
            return t
        raise ValueError(f"Invalid param shape: {tuple(t.shape)} expected (1,{dim}) or ({S},{dim})")
    raise ValueError(f"Invalid param ndim: {t.ndim}")

def build_cameras_from_vggt(extrinsic, intrinsic, H, W, device, args, verbose=False):
    camera_type = args.camera_type
    S = extrinsic.shape[0]
    ex = torch.from_numpy(extrinsic).to(device)
    K = torch.from_numpy(intrinsic).to(device)
    
    if ex.shape[1] == 3:
        ones = torch.ones((S, 1, 4), device=device)
        ones[:, :, :3] = 0
        w2c = torch.cat([ex, ones], dim=1)
    else:
        w2c = ex
        
    c2w = torch.inverse(w2c)
    
    R = c2w[:, :3, :3]
    T = c2w[:, :3, 3]
    R_map = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], dim=2)
    new_c2w = torch.cat([R_map, T[:, :, None]], dim=2)
    
    ones2 = torch.ones((S, 1, 4), device=device)
    ones2[:, :, :3] = 0
    c2w_aug = torch.cat([new_c2w, ones2], dim=1)
    w2c_new = torch.inverse(c2w_aug)
    
    R_new = w2c_new[:, :3, :3].permute(0, 2, 1).contiguous()
    T_new = w2c_new[:, :3, 3].contiguous()
    
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    
                                         
    scale_factor = 1.0
    if camera_type == "wide":
        scale_factor = 0.75
    elif camera_type == "ultra_wide":
        scale_factor = 0.5
    elif camera_type == "telephoto":
        scale_factor = 1.5
    elif camera_type == "fisheye":
                                                                       
        scale_factor = 1.0 
    elif camera_type == "ucm":
                                                                                      
        scale_factor = 1.0 
    elif camera_type == "dsm":
                            
        scale_factor = 1.0 

    fx = fx * scale_factor
    fy = fy * scale_factor

    cx = K[:, 0, 2] 
    cy = K[:, 1, 2]

    focal = torch.stack([fx, fy], dim=-1)
    pp = torch.stack([cx, cy], dim=-1)
    
    if camera_type == "fisheye":
        radial = _expand_param(FISHEYE_RADIAL, S, device, 6)
        tangential = _expand_param(FISHEYE_TANGENTIAL, S, device, 2)
        thin_prism = _expand_param(FISHEYE_THIN_PRISM, S, device, 4)
        cameras = FishEyeCameras(
            focal_length=focal,
            principal_point=pp,
            radial_params=radial,
            tangential_params=tangential,
            thin_prism_params=thin_prism,
            image_size=((H, W),),
            R=R_new,
            T=T_new,
            world_coordinates=True,
            device=device,
        )
    elif camera_type == "ucm":
        if UnifiedOmnidirectionalCameras is None:
            raise ImportError("UnifiedOmnidirectionalCameras is not available")
        
        xi_t = _expand_param(UCM_XI, S, device, 1)
        cameras = UnifiedOmnidirectionalCameras(
            focal_length=focal,
            principal_point=pp,
            xi=xi_t,
            image_size=((H, W),),
            R=R_new,
            T=T_new,
            world_coordinates=True,
            device=device,
        )
    elif camera_type == "dsm":
        if DoubleSphereCameras is None:
            raise ImportError("DoubleSphereCameras is not available")

        xi_t = _expand_param(DSM_XI, S, device, 1)
        alpha_t = _expand_param(DSM_ALPHA, S, device, 1)
        cameras = DoubleSphereCameras(
            focal_length=focal,
            principal_point=pp,
            xi=xi_t,
            alpha=alpha_t,
            image_size=((H, W),),
            R=R_new,
            T=T_new,
            world_coordinates=True,
            device=device,
            projection_mode="equirect",
        )
    else:
        cameras = PerspectiveCameras(
            focal_length=focal, principal_point=pp, in_ndc=False, 
            image_size=((H, W),), R=R_new, T=T_new, device=device
        )
    return cameras

def setup_renderer(cameras, image_size, radius, ppp=12):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, radius=radius, points_per_pixel=ppp, bin_size=0
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    return renderer, rasterizer

def filter_window_points(pts_t, rgb_t, fid_t, dyn_t, i, win_param, ex_row, K_row, H, W, device=None, camera_type=None):
    if isinstance(win_param, int):
        win_min = max(0, i - win_param)
        win_max = i + win_param
    else:
        win_min = max(0, i + win_param[0])
        win_max = i + win_param[1]
        
                                                                                 
    
                                                                        
    if camera_type in ["dsm", "ucm"]:
                                  
        fixed_win = 300
        win_min_fixed = max(0, i - fixed_win)
        win_max_fixed = i + fixed_win

                                                       
        sample_stride = 10
        sample_ids = [i - sample_stride, i, i + sample_stride]
                                                          
        sample_ids = [sid for sid in sample_ids if win_min_fixed <= sid <= win_max_fixed]
        if len(sample_ids) == 0:
            sample_ids = [i]
        sample_ids_t = torch.tensor(sample_ids, device=fid_t.device, dtype=fid_t.dtype)
        mask_static = (~dyn_t) & torch.isin(fid_t, sample_ids_t)
    else:
                                                               
        mask_static = (~dyn_t) & (fid_t >= win_min) & (fid_t <= win_max)

                                                          
    mask_dynamic = (dyn_t) & (fid_t == i)
    mask = mask_static | mask_dynamic
    
    p = pts_t[mask]
    c = rgb_t[mask]
    
    R = torch.from_numpy(ex_row[:3, :3]).to(p.device).float()
    t = torch.from_numpy(ex_row[:3, 3]).to(p.device).float()
    Kt = torch.from_numpy(K_row).to(p.device).float()
    
    P = torch.matmul(p, R.T) + t
    z = P[:, 2]

                                                         
    if camera_type in ["dsm", "ucm"]:
        return p, c
        
    m = z > 0
    if not torch.any(m):
        return p[:0], c[:0]
    P = P[m]
    c = c[m]
    fx, fy = Kt[0, 0], Kt[1, 1]
    cx, cy = Kt[0, 2], Kt[1, 2]
    u = fx * (P[:, 0] / z[m]) + cx
    v = fy * (P[:, 1] / z[m]) + cy
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    p = p[m][inside]
    c = c[inside]
    return p, c

def visualize_trajectories(C_o, C_s, out_path):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot(C_o[:, 0], C_o[:, 1], C_o[:, 2], color='blue', alpha=0.3, linewidth=1, label='Original')
    
    num_frames = C_s.shape[0]
    indices = np.arange(num_frames)
    sc = ax.scatter(C_s[:, 0], C_s[:, 1], C_s[:, 2], c=indices, cmap='viridis', s=5, label='Smoothed')
    ax.legend()
    
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()
    ranges = [abs(x_lim[1]-x_lim[0]), abs(y_lim[1]-y_lim[0]), abs(z_lim[1]-z_lim[0])]
    max_range = max(ranges)
    mid_x, mid_y, mid_z = np.mean(x_lim), np.mean(y_lim), np.mean(z_lim)
    ax.set_xlim3d(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim3d(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim3d(mid_z - max_range/2, mid_z + max_range/2)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(C_o[:, 0], C_o[:, 2], 'b-', alpha=0.3, label='Original')
    ax2.plot(C_s[:, 0], C_s[:, 2], 'r-', linewidth=1.5, label='Smoothed')
    ax2.set_title('Top View (X-Z)')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(C_o[:, 0], C_o[:, 1], 'b-', alpha=0.3, label='Original')
    ax3.plot(C_s[:, 0], C_s[:, 1], 'r-', linewidth=1.5, label='Smoothed')
    ax3.invert_yaxis() 
    ax3.set_title('Front View (X-Y)')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(C_o[:, 2], C_o[:, 1], 'b-', alpha=0.3, label='Original')
    ax4.plot(C_s[:, 2], C_s[:, 1], 'r-', linewidth=1.5, label='Smoothed')
    ax4.invert_yaxis()
    ax4.set_title('Side View (Z-Y)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def run_pass_1(model, paths, chunk_size, overlap, device):
    """
    Pass 1: Global Inference 1 - Scan video to get Depth and Raw Masks.
    """
    print("\n--- Pass 1: Global Inference 1 (Depth & Raw Masks) ---")
    step = max(1, chunk_size - overlap)
    num_frames = len(paths)
    
    results = {
        "depth": [],
        "intrinsic": [],
        "raw_mask": [],
        "raw_pose": [], 
        "images": [],
        "depth_conf": []
    }
    
    total_chunks = (num_frames + step - 1) // step
    for start_idx in tqdm(range(0, num_frames, step), total=total_chunks, desc="Pass 1"):
        end_idx = min(start_idx + chunk_size, num_frames)
        current_paths = paths[start_idx:end_idx]
        if len(current_paths) < 1: break
        
        images = load_and_preprocess_images(current_paths).to(device)
        n_img, _, h_img, w_img = images.shape
        
        with suppress_internal_tqdm(), suppress_output():
            predictions1, qk_dict, enc_feat, agg_tokens_list = inference(model, images)
            qk_dict = organize_qk_dict(qk_dict, images.shape[0])
            dyn_maps = extract_dyn_map(qk_dict, images)
        
        h_tok, w_tok = h_img // 14, w_img // 14
        feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)
        norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)
        
        upsampled_map = F.interpolate(rearrange(
            norm_dyn_map, "n_img h w -> n_img 1 h w"), size=(h_img, w_img), mode='bilinear', align_corners=False)
        upsampled_map = rearrange(upsampled_map, "n_img 1 h w -> n_img h w")
        
        thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
        dyn_masks = upsampled_map > thres
        
        results["depth"].append(predictions1["depth"]) 
        results["intrinsic"].append(predictions1["intrinsic"]) 
        results["raw_mask"].append(dyn_masks.cpu().numpy())
        results["raw_pose"].append(predictions1["cam2world"]) 
        results["images"].append(images.cpu().numpy())
        results["depth_conf"].append(predictions1.get("depth_conf", None))

        del predictions1, qk_dict, enc_feat, agg_tokens_list, dyn_maps, feat_map, norm_dyn_map, upsampled_map, dyn_masks, images
        torch.cuda.empty_cache()
        
        if end_idx == num_frames: break
        
    return results

def refine_masks_process(pass1_results, device):
    print("\n--- Refine Masks ---")
    refined_masks = []
    num_chunks = len(pass1_results["images"])
    
    for i in tqdm(range(num_chunks), total=num_chunks, desc="Refine Masks"):
        img = torch.from_numpy(pass1_results["images"][i]).to(device)
        depth = torch.from_numpy(pass1_results["depth"][i]).to(device)
        mask = torch.from_numpy(pass1_results["raw_mask"][i]).to(device)
        pose = torch.from_numpy(pass1_results["raw_pose"][i]).float().to(device)
        intr = torch.from_numpy(pass1_results["intrinsic"][i]).to(device)
        
        refiner = RefineDynMask(img, depth, mask, pose, intr, device)
        with suppress_internal_tqdm(), suppress_output():
            rm = refiner.refine_masks()
        
        if isinstance(rm, torch.Tensor):
            rm = rm.cpu().numpy()
        refined_masks.append(rm)
        del refiner, img, depth, mask, pose, intr
        torch.cuda.empty_cache()
        
    return refined_masks

def run_pass_2(model, pass1_results, refined_masks, device):
    """
    Pass 2: Global Inference 2 - Refined Pose using Refined Masks.
    """
    print("\n--- Pass 2: Global Inference 2 (Refined Pose) ---")
    
    results = {
        "pose": []
    }
    
    num_chunks = len(pass1_results["images"])
    
    for i in tqdm(range(num_chunks), total=num_chunks, desc="Pass 2"):
        images_np = pass1_results["images"][i]
        masks_np = refined_masks[i]
        
        images = torch.from_numpy(images_np).to(device)
        masks = torch.from_numpy(masks_np).to(device)
        
        with suppress_output():
            predictions2, _, _, _ = inference(model, images, masks)
        
        results["pose"].append(predictions2["cam2world"])
        
        del predictions2, images, masks
        torch.cuda.empty_cache()
        
    return results

def align_and_merge_chunks(pass1_results, pass2_results, refined_masks, overlap):
    """
    Stitch chunks into global trajectory.
    Aligns Pass 1 Depth and Pass 2 Pose.
    """
    
    all_depth = []
    all_pose = []
    all_intrinsic = []
    all_images = []
    all_masks = []
    
    num_chunks = len(pass1_results["depth"])
    
    prev_depth_chunk = None
    prev_pose_chunk = None
    
    for i in range(num_chunks):
        depth = pass1_results["depth"][i].copy()
        pose = pass2_results["pose"][i].copy()      
        mask_chunk = refined_masks[i].copy()
        
        if i > 0:
                                             
            n_ov = min(overlap, len(depth), len(prev_depth_chunk))
            
                                         
            prev_conf = pass1_results["depth_conf"][i-1] if pass1_results["depth_conf"][i-1] is not None else None
            curr_conf = pass1_results["depth_conf"][i] if pass1_results["depth_conf"][i] is not None else None
            
                                            
            pc = prev_conf[-n_ov:] if prev_conf is not None else None
            cc = curr_conf[:n_ov] if curr_conf is not None else None
            
            scale, T_align = robust_align_chunks(
                prev_depth_chunk[-n_ov:],
                prev_pose_chunk[-n_ov:],
                depth[:n_ov],
                pose[:n_ov],
                prev_conf=pc,
                curr_conf=cc
            )
                                                                              
            
            depth *= scale
            pose[:, :3, 3] *= scale
            pose = np.matmul(T_align[None, ...], pose)
            
                              
            if all_depth and n_ov > 0:
                 if len(all_depth[-1]) >= n_ov:
                                     
                     prev_d = all_depth[-1][-n_ov:]
                     curr_d = depth[:n_ov]
                     
                     alpha = np.linspace(0, 1, n_ov).astype(np.float32)
                     alpha_v = alpha[:, None, None] 
                     
                     blended_d = (1 - alpha_v) * prev_d + alpha_v * curr_d
                     all_depth[-1][-n_ov:] = blended_d
                     
                                    
                     prev_p = all_pose[-1][-n_ov:]
                     curr_p = pose[:n_ov]
                     
                     t1 = prev_p[:, :3, 3]
                     t2 = curr_p[:, :3, 3]
                     t_blend = (1 - alpha[:, None]) * t1 + alpha[:, None] * t2
                     
                     R1 = prev_p[:, :3, :3]
                     R2 = curr_p[:, :3, :3]
                     
                     q1 = Rotation.from_matrix(R1).as_quat()
                     q2 = Rotation.from_matrix(R2).as_quat()
                     
                     dots = np.sum(q1 * q2, axis=1)
                     q2[dots < 0] = -q2[dots < 0]
                     
                     q_blend = (1 - alpha[:, None]) * q1 + alpha[:, None] * q2
                     q_blend /= np.linalg.norm(q_blend, axis=1, keepdims=True)
                     R_blend = Rotation.from_quat(q_blend).as_matrix()
                     
                     pose_blend = np.eye(4)[None].repeat(n_ov, axis=0)
                     pose_blend[:, :3, :3] = R_blend
                     pose_blend[:, :3, 3] = t_blend
                     
                     all_pose[-1][-n_ov:] = pose_blend
                     
                                                  
                     prev_m = all_masks[-1][-n_ov:]
                     curr_m = mask_chunk[:n_ov]
                     merged_m = prev_m | curr_m
                     all_masks[-1][-n_ov:] = merged_m

        prev_depth_chunk = depth.copy()
        prev_pose_chunk = pose.copy()
        
        start_save = overlap if i > 0 else 0
        
        if start_save < len(depth):
            all_depth.append(depth[start_save:])
            all_pose.append(pose[start_save:])
            all_intrinsic.append(pass1_results["intrinsic"][i][start_save:])
            all_images.append(pass1_results["images"][i][start_save:])
            all_masks.append(mask_chunk[start_save:])
            
    global_depth = np.concatenate(all_depth, axis=0)
    global_pose = np.concatenate(all_pose, axis=0)
    global_intrinsic = np.concatenate(all_intrinsic, axis=0)
    global_images = np.concatenate(all_images, axis=0)
    global_masks = np.concatenate(all_masks, axis=0)
    
    return global_depth, global_pose, global_intrinsic, global_images, global_masks

def process_single_video(video_path, output_root, args):
    video_name = os.path.basename(video_path)
    rel_name_no_ext = os.path.splitext(video_name)[0]

                  
    device = select_device()
    
                       
    print("Loading VGGT4D model...")
    model = VGGTFor4D()
    
                               
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        print("Error: Checkpoint not found. Please specify --ckpt_path.")
        return
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
                                                                                 
             state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint not found. Please specify --ckpt_path.")
        return 

    model.eval()
    model = model.to(device)

    temp_work_dir = tempfile.mkdtemp(prefix=f"vs3r_{rel_name_no_ext}_")
    frames_dir = os.path.join(temp_work_dir, "frames")
    out_frames_dir = os.path.join(temp_work_dir, "stable_frames")

    out_video_path = os.path.join(output_root, f"{rel_name_no_ext}.mp4")
    
    try:
        print(f"\nProcessing: {video_name} (Three-Stage Pipeline)...")
        paths, (H0, W0), fps = extract_video_frames(video_path, args.frame_stride, frames_dir, max_frames=args.max_frames)
        if not paths: return
        
                                                        
        pass1_res = run_pass_1(model, paths, args.chunk_size, args.chunk_overlap, device)
        if not pass1_res["images"]: return
        
                      
        refined_masks = refine_masks_process(pass1_res, device)
        
                                                            
        pass2_res = run_pass_2(model, pass1_res, refined_masks, device)
        
                             
        g_depth, g_pose, g_intrinsic, g_images, g_masks = align_and_merge_chunks(pass1_res, pass2_res, refined_masks, args.chunk_overlap)
        
        opt_poses = g_pose
        
                                    
                                          
        preds = {
            "extrinsic": np.linalg.inv(opt_poses),             
            "intrinsic": g_intrinsic,
            "depth": g_depth,
            "images": g_images
        }
        
        preds["depth_conf"] = None
        
        render_H, render_W = preds["images"].shape[2], preds["images"].shape[3]

                                                      
        mask_to_use = g_masks
        raft_model, raft_transforms = load_raft_model(device)
        images_uint8 = (preds["images"].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        flow_mask = compute_dynamic_masks(
            images_uint8,
            preds["depth"],
            preds["intrinsic"],
            preds["extrinsic"],
            raft_model,
            raft_transforms,
            device,
            threshold=args.dynamic_threshold
        )
        mask_to_use = np.logical_or(mask_to_use, flow_mask)
        del raft_model
        torch.cuda.empty_cache()
        ratio = np.mean(mask_to_use) * 100
        print(f"  动态像素比例: {ratio:.2f}%")
        
        pts, rgb, fid, dyn_flags = aggregate_point_cloud(
            preds["depth"], preds["images"], preds["depth_conf"], 
            preds["extrinsic"], preds["intrinsic"], 
            dynamic_masks=mask_to_use,
            conf_thres=0.0
        )

                                            
        ex_s = smooth_camera_trajectory_gaussian(preds["extrinsic"], smooth_window=60, stability=10)
        
                                   
        pts_t = torch.from_numpy(pts).float().to(device)
        rgb_t = torch.from_numpy(rgb).float().to(device) / 255.0
        fid_t = torch.from_numpy(fid).to(device)
        dyn_t = torch.from_numpy(dyn_flags).bool().to(device)               
        
        rendered_temp_dir = out_frames_dir
        os.makedirs(rendered_temp_dir, exist_ok=True)
        g_min_x, g_min_y = render_W, render_H
        g_max_x, g_max_y = 0, 0
        has_valid_pixels = False
        
        common_ffmpeg_params = ['-crf', '18']
        
        total_frames = ex_s.shape[0]
        render_range = (-args.window_size, args.window_size)
        
                             
        tmp_cams = build_cameras_from_vggt(ex_s[0:1], preds["intrinsic"][0:1], render_H, render_W, device, args)
        renderer, rasterizer = setup_renderer(tmp_cams, (render_H, render_W), radius=args.radius, ppp=args.points_per_pixel)
        
        for i in tqdm(range(total_frames), total=total_frames, desc="Render"):
            curr_cams = build_cameras_from_vggt(ex_s[i:i+1], preds["intrinsic"][i:i+1], render_H, render_W, device, args)
            
                                              
            p_win, c_win = filter_window_points(
                pts_t, rgb_t, fid_t, dyn_t,
                i, render_range,
                ex_s[i], preds["intrinsic"][i],
                render_H, render_W,
                device,
                camera_type=args.camera_type
            )
            
            if p_win.shape[0] == 0:
                img_np = np.zeros((render_H, render_W, 3), dtype=np.uint8)
                mask_np_1ch = np.ones((render_H, render_W), dtype=np.uint8) * 255
            else:
                pc = Pointclouds(points=[p_win], features=[c_win])
                with torch.no_grad():
                    fragments = rasterizer(pc, cameras=curr_cams)
                    img_t = renderer(pc, cameras=curr_cams)
                    idx_map = fragments.idx[0, ..., 0]
                    mask_tensor = (idx_map == -1)
                    
                    mask_np_1ch = (mask_tensor.float() * 255).byte().cpu().numpy()
                    img_rgb = img_t[0, ..., :3]
                    img_rgb[mask_tensor] = 0.0
                    img_np = (img_rgb.clamp(0, 1) * 255).byte().cpu().numpy()
                    valid_mask = ~mask_tensor
                    if valid_mask.any():
                        y_inds, x_inds = torch.where(valid_mask)
                        y_min, y_max = y_inds.min().item(), y_inds.max().item()
                        x_min, x_max = x_inds.min().item(), x_inds.max().item()
                        g_min_x = min(g_min_x, x_min)
                        g_min_y = min(g_min_y, y_min)
                        g_max_x = max(g_max_x, x_max)
                        g_max_y = max(g_max_y, y_max)
                        has_valid_pixels = True
                    del img_t, pc, fragments, mask_tensor

            alpha_channel = 255 - mask_np_1ch
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_rgba = np.dstack([img_bgr, alpha_channel])
            cv2.imwrite(os.path.join(rendered_temp_dir, f"{i:06d}.png"), img_rgba)
            del curr_cams, p_win, c_win

        if not has_valid_pixels:
            print(f"[{video_name}] Warning: No valid pixels found. Using full size.")
            g_min_x, g_min_y = 0, 0
            g_max_x, g_max_y = render_W - 1, render_H - 1

        pad = 0
        crop_x1 = max(0, g_min_x - pad)
        crop_y1 = max(0, g_min_y - pad)
        crop_x2 = min(render_W - 1, g_max_x + pad)
        crop_y2 = min(render_H - 1, g_max_y + pad)

        crop_w = crop_x2 - crop_x1 + 1
        crop_h = crop_y2 - crop_y1 + 1

        if crop_w % 2 != 0: crop_w += 1
        if crop_h % 2 != 0: crop_h += 1

        crop_x2 = crop_x1 + crop_w - 1
        if crop_x2 >= render_W:
            diff = crop_x2 - (render_W - 1)
            crop_x1 = max(0, crop_x1 - diff)
            crop_x2 = render_W - 1
            crop_w = crop_x2 - crop_x1 + 1
            if crop_w % 2 != 0:
                crop_x2 -= 1
                crop_w -= 1

        crop_y2 = crop_y1 + crop_h - 1
        if crop_y2 >= render_H:
            diff = crop_y2 - (render_H - 1)
            crop_y1 = max(0, crop_y1 - diff)
            crop_y2 = render_H - 1
            crop_h = crop_y2 - crop_y1 + 1
            if crop_h % 2 != 0:
                crop_y2 -= 1
                crop_h -= 1

        print(f"[{video_name}] Global Crop Box: x=[{crop_x1}, {crop_x2}], y=[{crop_y1}, {crop_y2}], Size={crop_w}x{crop_h}")

        writer = imageio.get_writer(out_video_path, fps=fps, codec='libx264', pixelformat='yuv420p', ffmpeg_params=common_ffmpeg_params, macro_block_size=1)

        print(f"[{video_name}] Writing cropped video...")
        for i in range(total_frames):
            load_p = os.path.join(rendered_temp_dir, f"{i:06d}.png")
            if not os.path.exists(load_p): continue

            img_rgba = cv2.imread(load_p, cv2.IMREAD_UNCHANGED)
            if img_rgba is None: continue

            img_crop = img_rgba[crop_y1:crop_y2+1, crop_x1:crop_x2+1]
            if img_crop.size == 0: continue

            bgr = img_crop[..., :3]
            rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            writer.append_data(rgb_out)

        writer.close()
        shutil.rmtree(rendered_temp_dir, ignore_errors=True)
        del renderer, rasterizer, pts_t, rgb_t, fid_t, dyn_t
        torch.cuda.empty_cache()
        print(f"\n[{video_name}] 完成.")

    except Exception as e:
        print(f"[{video_name}] 处理出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import gc
        gc.collect()
        if os.path.exists(temp_work_dir): shutil.rmtree(temp_work_dir, ignore_errors=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_root", type=str, required=True, help="Root folder for outputs")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--window_size", type=int, default=1, help="Frame window for static points")
    parser.add_argument("--radius", type=float, default=0.012)
    parser.add_argument("--points_per_pixel", type=int, default=12)
    parser.add_argument("--fix_intrinsic", type=str2bool, default=True)
    parser.add_argument("--dynamic_threshold", type=float, default=0.5)
    
                     
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--chunk_overlap", type=int, default=10)
    
                       
    parser.add_argument("--ckpt_path", type=str, default="../ckpts/vggt4d/model_tracker_fixed_e20.pt")
    
                   
    parser.add_argument("--camera_type", type=str, default="perspective", choices=["perspective", "wide", "ultra_wide", "telephoto", "fisheye", "ucm", "dsm"])

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    os.makedirs(args.output_root, exist_ok=True)

    process_single_video(args.video_path, args.output_root, args)

if __name__ == "__main__":
    main()
