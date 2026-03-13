import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from sklearn.cluster import KMeans
from tqdm import tqdm


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


def grid_sample_depth(depths: torch.Tensor, uv: torch.Tensor):
    """
    depths: [n_img, 1, h_img, w_img]
    uv: [n_img, 1, n_pick, 2]
    """
    h, w = depths.shape[-2:]
    uv = uv[..., :2].clone()
    uv[..., 0] = uv[..., 0] / (w - 1)
    uv[..., 1] = uv[..., 1] / (h - 1)
    uv[..., 0] = uv[..., 0] * 2 - 1
    uv[..., 1] = uv[..., 1] * 2 - 1
    sample_depth = F.grid_sample(
        depths, uv, mode="nearest", align_corners=True)
    return sample_depth


def grid_sample_mask(masks: torch.Tensor, uv: torch.Tensor):
    """
    masks: [n_img, 1, h_img, w_img]
    uv: [n_img, 1, n_pick, 2]
    """
    masks = masks.float()
    h, w = masks.shape[-2:]
    uv = uv[..., :2].clone()
    uv[..., 0] = uv[..., 0] / (w - 1)
    uv[..., 1] = uv[..., 1] / (h - 1)
    uv[..., 0] = uv[..., 0] * 2 - 1
    uv[..., 1] = uv[..., 1] * 2 - 1
    sample_mask = F.grid_sample(
        masks, uv, mode="bilinear", align_corners=True)
    sample_mask = sample_mask > 0.5
    return sample_mask


def grid_sample_rgb(rgb: torch.Tensor, uv: torch.Tensor):
    """
    rgb: [n_img, 3, h_img, w_img]
    uv: [n_img, 1, n_pick, 2]
    """
    rgb = rgb.float()
    h, w = rgb.shape[-2:]
    uv = uv[..., :2].clone()
    uv[..., 0] = uv[..., 0] / (w - 1)
    uv[..., 1] = uv[..., 1] / (h - 1)
    uv[..., 0] = uv[..., 0] * 2 - 1
    uv[..., 1] = uv[..., 1] * 2 - 1
    sample_rgb = F.grid_sample(
        rgb, uv, mode="bilinear", align_corners=True)
    sample_rgb = sample_rgb
    return sample_rgb


class RefineDynMask:

    def __init__(self, images: torch.Tensor,
                 depths: torch.Tensor,
                 coarse_masks: torch.Tensor,
                 cam2world: torch.Tensor,
                 intrinsics: torch.Tensor,
                 device: torch.device):
        self.images = images
        self.coarse_masks = coarse_masks
        self.depths = depths
        self.cam2world = cam2world
        self.intrinsics = intrinsics
        self.device = device
        pts = inverse_project(self.depths, self.intrinsics, self.cam2world)
        self.pts = pts

    def _compute_dyn_loss(self, cam_id: int,
                          pts: torch.Tensor,
                          rgb: torch.Tensor,
                          labels: torch.Tensor,
                          dyn_labels: torch.Tensor):
        n_img, _, h_img, w_img = self.images.shape
        label_losses = []
        for label in dyn_labels:
            pick_mask = labels == label
            pick_pts = pts[pick_mask]
            pick_rgb = rgb[pick_mask]
            other_cam_id = torch.tensor(
                [i for i in range(n_img) if i != cam_id], dtype=torch.long)
            other_cam2world = self.cam2world[other_cam_id]
            other_world2cam = torch.inverse(other_cam2world)

            pick_pts = rearrange(pick_pts, "n_pick xyz -> n_pick xyz 1")
            pick_pts_cam = other_world2cam[:, None, :3, :3] @ pick_pts \
                + other_world2cam[:, None, :3, 3:4]
            other_K = self.intrinsics[other_cam_id]
            pick_pts_proj = other_K[:, None, ...] @ pick_pts_cam

            pick_pts_proj = pick_pts_proj[..., 0]
            pick_pts_proj[..., 0:2] = pick_pts_proj[..., 0:2] / \
                pick_pts_proj[..., 2:3]
            valid_width = (pick_pts_proj[..., 0] > 0) & (
                pick_pts_proj[..., 0] < w_img)
            valid_height = (pick_pts_proj[..., 1] > 0) & (
                pick_pts_proj[..., 1] < h_img)
            valid_depth = pick_pts_proj[..., 2] > 0
            valid_proj = valid_width & valid_height & valid_depth

            other_depths = self.depths[other_cam_id][:, None, ...]
            pick_pts_proj = rearrange(
                pick_pts_proj, "n_cam n_pick xyz -> n_cam 1 n_pick xyz")

            sample_depths = grid_sample_depth(other_depths, pick_pts_proj)

            other_dyn_masks = self.coarse_masks[other_cam_id][:, None, ...]
            sample_dyn_masks = grid_sample_mask(other_dyn_masks, pick_pts_proj)

            other_rgbs = self.images[other_cam_id]
            sample_rgbs = grid_sample_rgb(other_rgbs, pick_pts_proj)

            sample_depths = rearrange(
                sample_depths, "n_cam 1 1 n_pick -> n_cam n_pick")
            pick_pts_proj = rearrange(
                pick_pts_proj, "n_cam 1 n_pick xyz -> n_cam n_pick xyz")
            sample_dyn_masks = rearrange(
                sample_dyn_masks, "n_cam 1 1 n_pick -> n_cam n_pick")
            sample_rgbs = rearrange(
                sample_rgbs, "n_cam c 1 n_pick -> n_cam n_pick c")

            # 屏蔽不可见的点
            visible_mask = pick_pts_proj[..., 2] - 0.01 < sample_depths
            # visible and project to static area
            loss_mask = visible_mask & (~sample_dyn_masks)
            loss_mask = loss_mask & valid_proj

            num_loss_points = loss_mask.sum()
            total_sample_points = (n_img - 1) * pick_pts.shape[0]

            # 如果损失点太少，则认为这个label是动态的
            if (num_loss_points / (total_sample_points + 1e-6)) < 0.05:
                label_losses.append((label, 1e10, 1e10, 1e10))
                continue

            depth_diff = pick_pts_proj[..., 2] - sample_depths
            rgb_diff = pick_rgb.unsqueeze(0) - sample_rgbs
            valid_depth_diff = depth_diff[loss_mask]
            valid_rgb_diff = rgb_diff[loss_mask]
            valid_depth_diff = torch.abs(valid_depth_diff)
            valid_rgb_diff = torch.abs(valid_rgb_diff)
            valid_depth_diff = valid_depth_diff.sum()
            valid_rgb_diff = valid_rgb_diff.sum()
            depth_loss = valid_depth_diff / loss_mask.sum()
            rgb_loss = valid_rgb_diff / loss_mask.sum()
            total_loss = depth_loss + rgb_loss / 3

            label_losses.append((label, depth_loss, rgb_loss, total_loss))

        return label_losses

    @torch.no_grad()
    def _refine_mask(self, cam_id: int):
        n_img, _, h_img, w_img = self.images.shape
        pts = self.pts[cam_id]
        rgb = self.images[cam_id]
        pts = rearrange(pts, "h w xyz -> (h w) xyz")
        rgb = rearrange(rgb, "c h w -> (h w) c")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
        _, select_idx = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.5)
        # print(
        #     f"remove {pts.shape[0] - len(select_idx)} statistical outlier points")

        selected_mask = torch.zeros(pts.shape[0], dtype=torch.bool)
        selected_mask[select_idx] = True

        coarse_mask = self.coarse_masks[cam_id].cpu()
        coarse_mask = rearrange(coarse_mask, "h w -> (h w)")
        dyn_pts = pts[selected_mask & coarse_mask]

        n_clusters = 30
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        dyn_pts_labels = kmeans.fit_predict(dyn_pts.cpu().numpy())
        dyn_labels = np.unique(dyn_pts_labels)
        dyn_labels = torch.tensor(dyn_labels, dtype=torch.long)
        dyn_pts_labels = torch.tensor(dyn_pts_labels, dtype=torch.long)
        pts_labels = torch.zeros(pts.shape[0], dtype=torch.long)
        # -1 是静态，-2是离群点，>=0是动态
        pts_labels[selected_mask & (~coarse_mask)] = -1
        pts_labels[~selected_mask] = -2
        pts_labels[selected_mask & coarse_mask] = dyn_pts_labels

        label_losses = self._compute_dyn_loss(
            cam_id, pts, rgb, pts_labels, dyn_labels)

        thres = 0.1
        selected_labels = torch.tensor(
            [label for label, _, _, loss in label_losses if loss > thres])
        refine_dyn_mask = torch.isin(pts_labels, selected_labels)
        refine_dyn_mask = rearrange(refine_dyn_mask, "(h w) -> h w",
                                    h=h_img, w=w_img)
        return refine_dyn_mask

    def refine_masks(self):
        n_img = self.images.shape[0]
        refined_masks = []
        for i in tqdm(range(n_img)):
            mask = self._refine_mask(i)
            mask = mask.to(torch.uint8).cpu().numpy()
            mask = mask * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = mask > 0
            mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
            refined_masks.append(mask)
        refined_masks = torch.stack(refined_masks, dim=0)
        return refined_masks
