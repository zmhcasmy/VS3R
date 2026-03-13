import torch
from typing import List, Optional, Tuple, Union

from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import CamerasBase

DSM_SCALE = 1
DSM_XI = -0.2
DSM_ALPHA = 0.6
_focal_length = torch.tensor(((1.0, 1.0),))
_principal_point = torch.tensor(((0.0, 0.0),))
_xi = torch.tensor(((DSM_XI,),))
_alpha = torch.tensor(((DSM_ALPHA,),))
_R = torch.eye(3)[None]
_T = torch.zeros(1, 3)

EQUIRECT_H_FOV = 180.0
EQUIRECT_V_FOV = 90.0


class DoubleSphereCameras(CamerasBase):
    _FIELDS = (
        "focal_length",
        "principal_point",
        "xi",
        "alpha",
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
        alpha=_alpha,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        world_coordinates: bool = False,
        device: Device = "cpu",
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
        projection_mode: str = "dsm",
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

        self.projection_mode = projection_mode
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
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(self.device)
        if alpha.ndim == 0:
            alpha = alpha.view(1, 1)
        elif alpha.ndim == 1:
            alpha = alpha.view(-1, 1)
        self.alpha = alpha
        self.R = R.to(self.device)
        self.T = T.to(self.device)
        self.world_coordinates = world_coordinates
        self.epsilon = 1e-10

    def transform_points_screen(self, points, eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        if self.world_coordinates:
            P = self.get_world_to_view_transform().transform_points(points, eps=eps)
        else:
            P = points

        x = -P[..., 0] 
        y = P[..., 1]
        z = P[..., 2]

        if self.projection_mode == "equirect":
            norm = torch.sqrt(x**2 + y**2 + z**2 + self.epsilon)
            x_n = x / norm
            y_n = y / norm
            z_n = z / norm
            
            lon = torch.atan2(x_n, z_n)
            lat = torch.asin(torch.clamp(y_n, min=-1.0 + self.epsilon, max=1.0 - self.epsilon))
            h_fov_rad = torch.tensor(EQUIRECT_H_FOV, device=points.device).deg2rad()
            v_fov_rad = torch.tensor(EQUIRECT_V_FOV, device=points.device).deg2rad()
            
            if self.image_size is not None:
                H, W = self.image_size[0] if self.image_size.ndim == 2 else self.image_size
                
                u_norm = lon / (h_fov_rad / 2.0)
                v_norm = lat / (v_fov_rad / 2.0)
                
                u = (1.0 - (u_norm + 1.0) * 0.5) * (W - 1.0)
                
                v = (0.5 - v_norm * 0.5) * (H - 1.0)

            else:
                u = lon
                v = lat
            return torch.stack([u, v, z], dim=-1)

        d1 = torch.sqrt(x**2 + y**2 + z**2 + self.epsilon)
        
        xi = self.xi.view(-1, 1) if self.xi.shape[0] > 1 else self.xi.view(1, 1)
        alpha = self.alpha.view(-1, 1) if self.alpha.shape[0] > 1 else self.alpha.view(1, 1)
        
        z1 = xi * d1 + z
        d2 = torch.sqrt(x**2 + y**2 + z1**2 + self.epsilon)
        
        denom = alpha * d2 + (1 - alpha) * z1
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
