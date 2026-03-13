from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from vggt.layers.attention import Attention


class AttentionFor4D(Attention):
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def attention_with_dynamic_mask(self, q: Tensor, k: Tensor, v: Tensor,
                                    is_frame_attn: bool, layer_id: int, dyn_masks: Tensor) -> Tensor:

        B, H, S, dk = q.shape
        B, H, S, dv = v.shape
        B_img, S_img, HW = dyn_masks.shape

        pad = torch.zeros(5, dtype=torch.bool).to(q.device)
        pad = repeat(pad, "n -> b s n", b=B_img, s=S_img)
        dyn_masks = torch.cat([pad, dyn_masks], dim=-1)

        O = torch.empty_like(v)
        if is_frame_attn:
            # if frame attention
            print("Mask Frame Attention at layer", layer_id)
            dyn_masks = rearrange(dyn_masks, "b s n -> (b s) n")
            cam_idx = torch.tensor([0], dtype=torch.long)
            rest_idx = torch.arange(1, S)
        else:
            # if global attention
            print("Mask Global Attention at layer", layer_id)
            dyn_masks = rearrange(dyn_masks, "b s n -> b (s n)")
            cam_idx = torch.arange(0, S, S // S_img)
            all_idx = torch.arange(S)
            cam_mask = torch.ones(S, dtype=torch.bool)
            cam_mask[cam_idx] = False
            rest_idx = all_idx[cam_mask]

        for b in range(B):
            qb = q[b:b+1]
            kb = k[b:b+1].contiguous()
            vb = v[b:b+1].contiguous()

            dyn_mask = dyn_masks[b]
            non_dyn_idx = (~dyn_mask).nonzero(as_tuple=True)[0]

            non_dyn_k = kb[..., non_dyn_idx, :].contiguous()
            non_dyn_v = vb[..., non_dyn_idx, :].contiguous()

            o = F.scaled_dot_product_attention(qb, non_dyn_k, non_dyn_v)
            O[b:b+1] = o

        return O

    def forward(self, x: Tensor, pos=None, is_frame_attn: bool = True, layer_id: int = 0, dyn_masks: Tensor = None) -> Tuple[Tensor, Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            need_mask_atten = True
            need_mask_atten = need_mask_atten and dyn_masks is not None
            need_mask_atten = need_mask_atten and layer_id in range(0, 5)
            if need_mask_atten:
                x = self.attention_with_dynamic_mask(
                    q, k, v, is_frame_attn, layer_id, dyn_masks)
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, q, k
