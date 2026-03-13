from typing import Tuple

from vggt4d.layers.attention import AttentionFor4D
from torch import Tensor

from vggt.layers.block import Block, drop_add_residual_stochastic_depth


class BlockFor4D(Block):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        kwargs["attn_class"] = AttentionFor4D
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, pos=None, is_frame_attn: bool = True, layer_id: int = 0, dyn_masks: Tensor = None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None) -> Tuple[Tensor, Tensor]:
            x, q, k = self.attn(self.norm1(
                x), pos=pos, is_frame_attn=is_frame_attn, layer_id=layer_id, dyn_masks=dyn_masks)
            x = self.ls1(x)
            return x, q, k

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            attn_x, attn_q, attn_k = attn_residual_func(x, pos=pos)
            x = x + self.drop_path1(attn_x)
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            attn_x, attn_q, attn_k = attn_residual_func(x, pos=pos)
            x = x + attn_x
            x = x + ffn_residual_func(x)
        return x, attn_q, attn_k
