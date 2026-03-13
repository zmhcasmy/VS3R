from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from vggt4d.layers.block import BlockFor4D
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten


class AggregatorFor4D(Aggregator):
    def __init__(self, **kwargs):
        kwargs["block_fn"] = BlockFor4D
        super().__init__(**kwargs)

    def forward(self, images: torch.Tensor,
                dyn_masks: Optional[torch.Tensor] = None,
                enable_memory_saving: bool = True) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            dyn_masks (torch.Tensor): Dynamic masks with shape [B, S, H, W], in range [0, 1].

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if dyn_masks is not None:
            dyn_masks = F.max_pool2d(
                dyn_masks.float(), kernel_size=self.patch_size, stride=self.patch_size)
            dyn_masks = rearrange(dyn_masks, "b s h w -> b s (h w)") > 0.5
            # dyn_masks[:, 0] = False
            # set patch tokens to 0 if dyn_masks is true
            # bad effect
            # print("Masking patch tokens")
            # patch_tokens[rearrange(dyn_masks, "b s n -> (b s) n")] = 0

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = [None] * (self.aa_block_num * B)
        global_q_list = []
        frame_q_list = []
        global_k_list = []
        frame_k_list = []
        preserve_layer_idx = [4, 11, 17, 23]

        for i in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates, frame_q, frame_k = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, dyn_masks=dyn_masks,
                    )
                    frame_q_list.append(frame_q.detach().cpu())
                    frame_k_list.append(frame_k.detach().cpu())
                    del frame_q, frame_k
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates, global_q, global_k = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, dyn_masks=dyn_masks,
                    )
                    global_q_list.append(global_q.detach().cpu())
                    global_k_list.append(global_k.detach().cpu())
                    del global_q, global_k
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for j in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat(
                    [frame_intermediates[j], global_intermediates[j]], dim=-1)
                output_list[i * B + j] = concat_inter

            if enable_memory_saving:
                if i not in preserve_layer_idx:
                    for j in range(B):
                        tmp = output_list[i * B + j]
                        output_list[i * B + j] = None
                        del tmp
                del concat_inter, frame_intermediates, global_intermediates

        global_q = torch.stack(global_q_list, dim=0)
        global_k = torch.stack(global_k_list, dim=0)
        frame_q = torch.stack(frame_q_list, dim=0)
        frame_k = torch.stack(frame_k_list, dim=0)

        if enable_memory_saving:
            del tokens

        qk_dict = {
            "global_q": global_q,
            "global_k": global_k,
            "frame_q": frame_q,
            "frame_k": frame_k
        }

        if "concat_inter" in locals():
            del concat_inter
        if "frame_intermediates" in locals():
            del frame_intermediates
        if "global_intermediates" in locals():
            del global_intermediates

        if enable_memory_saving:
            self.clear_inference_cache()

        return output_list, self.patch_start_idx, qk_dict, patch_tokens

    def clear_inference_cache(self):
        if hasattr(self, "rope") and self.rope is not None:
            if hasattr(self.rope, "frequency_cache"):
                self.rope.frequency_cache.clear()
        if hasattr(self, "position_getter") and self.position_getter is not None:
            if hasattr(self.position_getter, "position_cache"):
                self.position_getter.position_cache.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, dyn_masks: Optional[torch.Tensor] = None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        attn_q = []
        attn_k = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens, q, k = checkpoint(
                    self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens, q, k = self.frame_blocks[frame_idx](
                    tokens, pos=pos, is_frame_attn=True, layer_id=frame_idx, dyn_masks=dyn_masks)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            attn_q.append(q)
            attn_k.append(k)

        attn_q = torch.stack(attn_q, dim=0)
        attn_k = torch.stack(attn_k, dim=0)
        return tokens, frame_idx, intermediates, attn_q, attn_k

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, dyn_masks: Optional[torch.Tensor] = None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        attn_q = []
        attn_k = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens, q, k = checkpoint(
                    self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens, q, k = self.global_blocks[global_idx](
                    tokens, pos=pos, is_frame_attn=False, layer_id=global_idx, dyn_masks=dyn_masks)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            attn_q.append(q)
            attn_k.append(k)

        attn_q = torch.stack(attn_q, dim=0)
        attn_k = torch.stack(attn_k, dim=0)
        return tokens, global_idx, intermediates, attn_q, attn_k
