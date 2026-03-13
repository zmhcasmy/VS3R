import torch
from typing import List, Optional, Tuple, Union

from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import CamerasBase

UCM_SCALE = 2
UCM_XI = 1.0
_focal_length = torch.tensor(((1.0, 1.0),))
_principal_point = torch.tensor(((0.0, 0.0),))
_xi = torch.tensor(((UCM_XI,),))
_R = torch.eye(3)[None]
_T = torch.zeros(1, 3)


class UnifiedOmnidirectionalCameras(CamerasBase):
    _FIELDS = (
        "focal_length",
        "principal_point",
        "xi",
        "R",
        "T",
        "world_coordinates",
        "device",
        "image_size",
    )

    def __init__(
        self,
        focal_length=_focal_length,
        principal_point=_principal_point,
        xi=_xi,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        world_coordinates: bool = False,
        device: Device = "cpu",
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> None:
        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            R=R,
            T=T,
            **kwargs,
        )
        if image_size is not None:
            if (self.image_size < 1).any():
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        self.device = device
        self.focal = focal_length.to(self.device)
        self.principal_point = principal_point.to(self.device)
        if not torch.is_tensor(xi):
            xi = torch.tensor(xi)
        xi = xi.to(self.device)
        if xi.ndim == 0:
            xi = xi.view(1, 1)
        elif xi.ndim == 1:
            xi = xi.view(-1, 1)
        self.xi = xi
        self.R = R.to(self.device)
        self.T = T.to(self.device)
        self.world_coordinates = world_coordinates
        self.epsilon = 1e-10

    def transform_points_screen(self, points, eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        if self.world_coordinates:
            P = self.get_world_to_view_transform().transform_points(points, eps=eps)
        else:
            P = points
            
        x = P[..., 0]
        y = -P[..., 1]
        z = P[..., 2]

        d = torch.sqrt(x**2 + y**2 + z**2 + self.epsilon)
        
        xi = self.xi.view(-1, 1) if self.xi.shape[0] > 1 else self.xi.view(1, 1)
        
        denom = xi * d + z
        denom = torch.clamp(denom, min=self.epsilon)
        
        x_norm = x / denom
        y_norm = y / denom
        
        fx = self.focal[..., 0].unsqueeze(1)
        fy = self.focal[..., 1].unsqueeze(1)
        cx = self.principal_point[..., 0].unsqueeze(1)
        cy = self.principal_point[..., 1].unsqueeze(1)
        
        u = fx * x_norm + cx
        v = fy * y_norm + cy
        
        return torch.stack([u, v, z], dim=-1)

    def transform_points(self, points, eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        screen_points = self.transform_points_screen(points, eps, **kwargs)
        u, v, z = screen_points.unbind(-1)

        if self.image_size is not None:
            H, W = self.image_size[0] if self.image_size.ndim == 2 else self.image_size
            u_ndc = (2.0 * u / (W - 1.0)) - 1.0
            v_ndc = 1.0 - (2.0 * v / (H - 1.0))
            return torch.stack([u_ndc, v_ndc, z], dim=-1)

        return screen_points

    def in_ndc(self):
        return self.image_size is not None

    def is_perspective(self):
        return False
