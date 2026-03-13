import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import viser.transforms as tf
from einops import einsum, rearrange, repeat
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import cv2


def inverse_project(depth: torch.Tensor,
                    intrinsics: torch.Tensor,
                    cam2world: torch.Tensor):
    """
    depth: [n_img, h_img, w_img]
    intrinsics: [n_img, 3, 3]
    cam2world: [n_img, 4, 4]
    return: [n_img, h_img, w_img, 3]
    """
    n_img, h_img, w_img = depth.shape
    y, x = torch.meshgrid(torch.arange(
        h_img), torch.arange(w_img), indexing="ij")
    y = y.to(depth.device) + 0.5
    x = x.to(depth.device) + 0.5
    y = y.unsqueeze(0).expand(n_img, -1, -1)
    x = x.unsqueeze(0).expand(n_img, -1, -1)
    xyz = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    xyz = xyz * depth.unsqueeze(-1)
    xyz = rearrange(xyz, "n_img h w xyz -> h w n_img xyz 1")
    xyz = torch.inverse(intrinsics) @ xyz
    xyz = cam2world[..., :3, :3] @ xyz + cam2world[..., :3, 3, None]
    xyz = rearrange(xyz, "h w n_img xyz 1 -> n_img h w xyz")
    return xyz


def filter_outliers(pts: np.ndarray):
    filter_masks = np.zeros(pts.shape[0], dtype=np.bool_)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _, select_idx = pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.5)
    filter_masks[select_idx] = True
    return filter_masks


class Scene4D:
    def __init__(self, scene_dir: Path):
        self.scene_dir = scene_dir

        print("Loading confidence maps...")
        conf_paths = sorted(scene_dir.glob("conf_*.npy"), key=lambda x: int(x.stem.split("_")[-1]))
        confs = np.array([np.load(p) for p in tqdm(conf_paths, unit="frame")])
        self.confs = confs

        print("Loading images...")
        img_paths = sorted(scene_dir.glob("frame_*.png"), key=lambda x: int(x.stem.split("_")[-1]))
        # img_paths = sorted(scene_dir.glob("frame_*.jpg"))
        images = np.array([np.array(Image.open(p))
                           for p in tqdm(img_paths, unit="frame")])
        images = images.astype(np.float32) / 255.0
        rgb = rearrange(images, "n h w c -> n (h w) c").copy()
        self.images = images
        self.rgb = rgb

        print("Loading depths...")
        depth_paths = sorted(scene_dir.glob("frame_*.npy"), key=lambda x: int(x.stem.split("_")[-1]))
        depths = np.array([np.load(p)
                          for p in tqdm(depth_paths, unit="frame")])
        self.depths = depths

        dyn_mask_paths = sorted(scene_dir.glob("dynamic_mask_*.png"), key=lambda x: int(x.stem.split("_")[-1]))
        print("Loading dynamic masks...")
        dyn_masks = np.array([np.array(Image.open(p))
                             for p in tqdm(dyn_mask_paths, unit="frame")])
        if dyn_masks.shape[0] == 0:
            dyn_masks = np.ones((images.shape[0], images.shape[1], images.shape[2]), dtype=np.bool_)
        dyn_masks = rearrange(dyn_masks > 0, "n h w -> n (h w)")
        dyn_masks = dyn_masks.astype(np.uint8)
        kernel = np.ones((1, 1), np.uint8)
        dyn_masks = cv2.erode(dyn_masks, kernel, iterations=1)
        # dyn_masks = cv2.dilate(dyn_masks, kernel, iterations=1)
        dyn_masks = dyn_masks > 0
        self.dyn_masks = dyn_masks

        tum_poses = np.loadtxt(scene_dir / "pred_traj.txt")
        R = Rotation.from_quat(tum_poses[:, 4:8], scalar_first=True)
        T = tum_poses[:, 1:4]
        cam2world = np.concatenate([R.as_matrix(), T[:, :, None]], axis=-1)
        pad = np.zeros((cam2world.shape[0], 1, 4))
        pad[:, 0, -1] = 1.0
        cam2world = np.concatenate([cam2world, pad], axis=1)
        self.cam2world = cam2world

        intrinsics = np.loadtxt(scene_dir / "pred_intrinsics.txt")
        intrinsics = rearrange(intrinsics, "n (h w) -> n h w", h=3, w=3)
        self.intrinsics = intrinsics

        pts = inverse_project(torch.tensor(depths).float(),
                              torch.tensor(intrinsics).float(),
                              torch.tensor(cam2world).float()).cpu().numpy()
        pts = rearrange(pts, "n h w xyz -> n (h w) xyz")
        self.pts = pts

        # print("Filtering outliers...")
        # filter_masks = np.zeros(pts.shape[0:2], dtype=np.bool_)
        # for i in tqdm(range(pts.shape[0]), unit="frame"):
        #     filter_masks[i] = filter_outliers(pts[i])

        # select_masks = np.logical_and(filter_masks, ~dyn_masks)
        # filter_masks2 = filter_outliers(pts[select_masks])
        # filter_masks[select_masks] = np.logical_and(
        #     filter_masks[select_masks], filter_masks2)
        # self.filter_masks = filter_masks

    @property
    def num_frame(self):
        return self.images.shape[0]

    @property
    def num_point(self):
        return self.pts.shape[0] * self.pts.shape[1]

    @property
    def num_static_point(self):
        return np.sum(~self.dyn_masks)
        # return np.sum(np.logical_and(self.filter_masks, ~self.dyn_masks))

    @property
    def num_dynamic_point(self):
        return np.sum(self.dyn_masks)
        # return np.sum(np.logical_and(self.filter_masks, self.dyn_masks))

    def get_background_points(self, frame_id: int = None, filter: bool = True, downsample_factor: float = 1.0):
        if frame_id is not None:
            bg_masks = ~self.dyn_masks[frame_id]
            bg_pts = self.pts[frame_id][bg_masks]
            bg_rgb = self.rgb[frame_id][bg_masks]
        else:
            bg_masks = ~self.dyn_masks
            bg_pts = self.pts[bg_masks]
            bg_rgb = self.rgb[bg_masks]
        if bg_pts.shape[0] == 0:
            return bg_pts, bg_rgb
        if filter:
            filter_masks = filter_outliers(bg_pts)
            bg_pts = bg_pts[filter_masks]
            bg_rgb = bg_rgb[filter_masks]
        if downsample_factor < 1.0:
            idx = np.random.choice(bg_pts.shape[0], size=int(
                len(bg_pts) * downsample_factor), replace=False)
            return bg_pts[idx], bg_rgb[idx]
        else:
            return bg_pts, bg_rgb

    def get_dynamic_points(self, frame_id: int, downsample_factor: float = 1.0):
        dyn_masks = self.dyn_masks[frame_id]
        dyn_pts = self.pts[frame_id][dyn_masks]
        dyn_rgb = self.rgb[frame_id][dyn_masks]
        filter_masks = filter_outliers(dyn_pts)
        dyn_pts = dyn_pts[filter_masks]
        dyn_rgb = dyn_rgb[filter_masks]
        if downsample_factor < 1.0:
            idx = np.random.choice(dyn_pts.shape[0], size=int(
                len(dyn_pts) * downsample_factor), replace=False)
            return dyn_pts[idx], dyn_rgb[idx]
        else:
            return dyn_pts, dyn_rgb

    def get_cam_frustum(self, frame_id: int):
        cam2world = self.cam2world[frame_id]
        intrinsics = self.intrinsics[frame_id]
        h_img, w_img = self.images[frame_id].shape[:2]
        fx = intrinsics[0, 0]
        fov = 2 * np.arctan2(h_img / 2, fx)
        aspect = w_img / h_img
        wxyz = tf.SO3.from_matrix(cam2world[:3, :3]).wxyz
        pos = cam2world[:3, 3]
        return fov, aspect, wxyz, pos
