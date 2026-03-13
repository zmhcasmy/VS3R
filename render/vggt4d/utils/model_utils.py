import numpy as np
import torch
from einops import rearrange

from vggt4d.models.vggt4d import VGGTFor4D
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def inference(model: VGGTFor4D, images: torch.Tensor, dyn_masks: torch.Tensor = None, query_points: torch.Tensor = None) -> tuple[dict, dict, dict, list]:
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions, qk_dict, enc_feat, agg_tokens_list = model(
                images, dyn_masks=dyn_masks, query_points=query_points)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].to(device="cpu", dtype=torch.float32) \
                .numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # save memory intermediate aggregated tokens for tracking
    for i in range(len(agg_tokens_list)):
        if i not in [4, 11, 17, 23]:
            agg_tokens_list[i] = None

    torch.cuda.empty_cache()

    n_img = images.shape[0]
    pred_extrinsic = predictions["extrinsic"]
    pad = np.zeros((n_img, 1, 4))
    pad[:, 0, -1] = 1
    pred_extrinsic = np.concatenate([pred_extrinsic, pad], axis=1)
    pred_cam2world = np.linalg.inv(pred_extrinsic)
    predictions["cam2world"] = pred_cam2world
    predictions["depth"] = predictions["depth"].squeeze(-1)
    return predictions, qk_dict, enc_feat.detach().cpu(), agg_tokens_list


def organize_qk_dict(qk_dict, n_img):
    global_q = qk_dict["global_q"]
    global_k = qk_dict["global_k"]
    frame_q = qk_dict["frame_q"]
    frame_k = qk_dict["frame_k"]

    n_tok = global_q.shape[-2] // n_img

    patch_start_idx = 5

    global_q = rearrange(
        global_q, "n_layer 1 1 n_head (n_img n_tok) c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)
    global_k = rearrange(
        global_k, "n_layer 1 1 n_head (n_img n_tok) c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)

    global_cam_q = global_q[..., 0:1, :]
    global_cam_k = global_k[..., 0:1, :]
    global_reg_q = global_q[..., 1:patch_start_idx, :]
    global_reg_k = global_k[..., 1:patch_start_idx, :]
    global_tok_q = global_q[..., patch_start_idx:, :]
    global_tok_k = global_k[..., patch_start_idx:, :]

    frame_q = rearrange(
        frame_q, "n_layer 1 n_img n_head n_tok c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)
    frame_k = rearrange(
        frame_k, "n_layer 1 n_img n_head n_tok c -> n_img n_layer n_head n_tok c", n_img=n_img, n_tok=n_tok)

    frame_cam_q = frame_q[..., 0:1, :]
    frame_cam_k = frame_k[..., 0:1, :]
    frame_reg_q = frame_q[..., 1:patch_start_idx, :]
    frame_reg_k = frame_k[..., 1:patch_start_idx, :]
    frame_tok_q = frame_q[..., patch_start_idx:, :]
    frame_tok_k = frame_k[..., patch_start_idx:, :]

    return {
        "global_cam_q": global_cam_q,
        "global_cam_k": global_cam_k,
        "global_reg_q": global_reg_q,
        "global_reg_k": global_reg_k,
        "global_tok_q": global_tok_q,
        "global_tok_k": global_tok_k,
        "frame_cam_q": frame_cam_q,
        "frame_cam_k": frame_cam_k,
        "frame_reg_q": frame_reg_q,
        "frame_reg_k": frame_reg_k,
        "frame_tok_q": frame_tok_q,
        "frame_tok_k": frame_tok_k,

        "global_q": global_tok_q,
        "global_k": global_tok_k,
        "frame_q": frame_tok_q,
        "frame_k": frame_tok_k,
    }
