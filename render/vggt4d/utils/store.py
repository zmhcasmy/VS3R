from pathlib import Path

import cv2
from einops import rearrange
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface, plot
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch
from torchvision.utils import save_image
from copy import deepcopy


def save_dynamic_masks(data_dir, masks):
    for i, dynamic_mask in enumerate(masks):
        img_path = data_dir / f"dynamic_mask_{i:04d}.png"
        cv2.imwrite(img_path, (dynamic_mask *
                    255).detach().cpu().numpy().astype(np.uint8))


def save_intrinsic_txt(data_dir, intrinsic):
    intrinsic = rearrange(intrinsic, "n_img h w -> n_img (h w)")
    np.savetxt(data_dir / "pred_intrinsics.txt", intrinsic, fmt="%f")


def save_rgb(data_dir, images):
    n_img = images.shape[0]
    for i in range(n_img):
        save_image(images[i], data_dir / f"frame_{i:04d}.png")


def save_depth(data_dir, depths):
    if depths is torch.Tensor:
        depths = depths.cpu().numpy()
    n_img = depths.shape[0]
    for i in range(n_img):
        np.save(data_dir / f"frame_{i:04d}.npy", depths[i])


def save_depth_conf(data_dir, conf):
    if conf is torch.Tensor:
        conf = conf.cpu().numpy()
    n_img = conf.shape[0]
    for i in range(n_img):
        np.save(data_dir / f"conf_{i:04d}.npy", conf[i])


def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation

    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    if c2w is torch.Tensor:
        c2w = c2w.cpu().numpy()
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple) or isinstance(args, list):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def to_tum_poses(c2ws):
    if c2ws is torch.Tensor:
        c2ws = c2ws.cpu().numpy()

    tt = np.arange(c2ws.shape[0]).astype(float)
    tum_poses = [c2w_to_tumpose(c) for c in c2ws]
    tum_poses = np.stack(tum_poses, 0)
    traj = [tum_poses, tt]
    return traj


def save_tum_poses(data_dir, c2ws):
    if c2ws is torch.Tensor:
        c2ws = c2ws.cpu().numpy()

    tt = np.arange(c2ws.shape[0]).astype(float)
    tum_poses = [c2w_to_tumpose(c) for c in c2ws]
    tum_poses = np.stack(tum_poses, 0)
    traj = [tum_poses, tt]
    traj = make_traj(traj)
    def tostr(a): return " ".join(map(str, a))
    with (data_dir / "pred_traj.txt").open("w") as f:
        for i in range(traj.num_poses):
            f.write(
                f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[0, 1, 2, 3]])}\n"
            )


def load_tum_poses(data_dir):
    data = np.loadtxt(data_dir / "pred_traj.txt")
    pred_pose = np.zeros((data.shape[0], 4, 4))
    pred_pose[:, :3, 3] = data[:, 1:4]
    pred_pose[:, :3, :3] = Rotation.from_quat(
        data[:, 4:], scalar_first=True).as_matrix()
    pred_pose[:, 3, 3] = 1.0
    pred_pose = pred_pose.astype(np.float32)
    return pred_pose


def save_dynamic_masks(data_dir, masks):
    for i, dynamic_mask in enumerate(masks):
        img_path = data_dir / f"dynamic_mask_{i:04d}.png"
        cv2.imwrite(img_path, (dynamic_mask *
                    255).detach().cpu().numpy().astype(np.uint8))


def enlarge_seg_masks(data_dir, kernel_size=5):
    dyn_mask_paths = list(data_dir.glob("dynamic_mask_*.png"))
    dyn_mask_paths = sorted(dyn_mask_paths)
    for mask_path in dyn_mask_paths:
        id = int(mask_path.stem.split("_")[-1])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        enlarged_mask = cv2.dilate(mask, kernel, iterations=1)
        save_path = data_dir / f"enlarged_dynamic_mask_{id:04d}.png"
        cv2.imwrite(save_path, enlarged_mask)


def save_pts_ply(data_dir: Path, pts: np.ndarray, rgb: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    ply_path = data_dir / "points.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ply_path.absolute()), pcd)


def save_vggt4d_result(data_dir: Path,
                       cam2world: np.ndarray,
                       intrinsic: np.ndarray,
                       images: np.ndarray,
                       depth: np.ndarray,
                       conf: np.ndarray,
                       dyn_masks: np.ndarray = None):
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "cam2world.npy", cam2world)
    np.save(data_dir / "intrinsic.npy", intrinsic)
    np.save(data_dir / "images.npy", images)
    np.save(data_dir / "depth.npy", depth)
    np.save(data_dir / "conf.npy", conf)
    if dyn_masks is not None:
        np.save(data_dir / "dyn_masks.npy", dyn_masks)


def load_vggt4d_result(data_dir: Path):
    cam2world = np.load(data_dir / "cam2world.npy")
    intrinsic = np.load(data_dir / "intrinsic.npy")
    images = np.load(data_dir / "images.npy")
    depth = np.load(data_dir / "depth.npy")
    conf = np.load(data_dir / "conf.npy")
    if (data_dir / "dyn_masks.npy").exists():
        dyn_masks = np.load(data_dir / "dyn_masks.npy")
    else:
        dyn_masks = None
    return cam2world, intrinsic, images, depth, conf, dyn_masks
