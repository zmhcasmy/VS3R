import math
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import CamerasBase

FISHEYE_SCALE = 0.9
FISHEYE_RADIAL = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FISHEYE_TANGENTIAL = [0.0, 0.0]
FISHEYE_THIN_PRISM = [0.0, 0.0, 0.0, 0.0]
_focal_length = torch.tensor(((1.0,),))
_principal_point = torch.tensor(((0.0, 0.0),))
_radial_params = torch.tensor((tuple(FISHEYE_RADIAL),))
_tangential_params = torch.tensor((tuple(FISHEYE_TANGENTIAL),))
_thin_prism_params = torch.tensor((tuple(FISHEYE_THIN_PRISM),))

_R = torch.eye(3)[None]
_T = torch.zeros(1, 3)

class FishEyeCameras(CamerasBase):
    _FIELDS = (
        "focal_length",
        "principal_point",
        "R",
        "T",
        "radial_params",
        "tangential_params",
        "thin_prism_params",
        "world_coordinates",
        "use_radial",
        "use_tangential",
        "use_thin_prism",
        "device",
        "image_size",
    )

    def __init__(
        self,
        focal_length=_focal_length,
        principal_point=_principal_point,
        radial_params=_radial_params,
        tangential_params=_tangential_params,
        thin_prism_params=_thin_prism_params,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        world_coordinates: bool = False,
        use_radial: bool = True,
        use_tangential: bool = True,
        use_thin_prism: bool = True,
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
        self.radial_params = radial_params.to(self.device)
        self.tangential_params = tangential_params.to(self.device)
        self.thin_prism_params = thin_prism_params.to(self.device)
        self.R = R.to(self.device)
        self.T = T.to(self.device)
        self.world_coordinates = world_coordinates
        self.use_radial = use_radial
        self.use_tangential = use_tangential
        self.use_thin_prism = use_thin_prism
        self.epsilon = 1e-10
        self.num_distortion_iters = 50
        self.num_radial = radial_params.shape[-1]

    def transform_points_screen(self, points, eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        if self.world_coordinates:
            P = self.get_world_to_view_transform().transform_points(points, eps=eps)
        else:
            P = points

        x, y, z = P.unbind(-1)
        x_img = x
        y_img = -y
        
        r_xy = torch.sqrt(x_img**2 + y_img**2 + self.epsilon)
        theta = torch.atan2(r_xy, z)
        
        if self.use_radial:
            theta2 = theta**2
            theta_powers = torch.stack([theta2**(i+1) for i in range(self.radial_params.shape[-1])], dim=-1)
            
            if self.radial_params.ndim == 2 and theta_powers.ndim == 2:
                scaling = 1.0 + torch.sum(self.radial_params * theta_powers, dim=-1)
            else:
                scaling = 1.0 + torch.sum(self.radial_params * theta_powers, dim=-1)
                
            theta_d = theta * scaling
        else:
            theta_d = theta

        scale = theta_d / r_xy
        scale = torch.where(r_xy < 1e-6, torch.ones_like(scale), scale)
        
        u_distorted = x_img * scale
        v_distorted = y_img * scale

        if self.use_tangential:
            p1 = self.tangential_params[..., 0]
            p2 = self.tangential_params[..., 1]
            if p1.ndim == 1: p1 = p1.unsqueeze(1)
            if p2.ndim == 1: p2 = p2.unsqueeze(1)
            
            r2 = u_distorted**2 + v_distorted**2
            du = 2*p1*u_distorted*v_distorted + p2*(r2 + 2*u_distorted**2)
            dv = p1*(r2 + 2*v_distorted**2) + 2*p2*u_distorted*v_distorted
            u_distorted = u_distorted + du
            v_distorted = v_distorted + dv

        fx = self.focal[..., 0].unsqueeze(1)
        fy = self.focal[..., 1].unsqueeze(1)
        cx = self.principal_point[..., 0].unsqueeze(1)
        cy = self.principal_point[..., 1].unsqueeze(1)

        u = fx * u_distorted + cx
        v = fy * v_distorted + cy
        
        return torch.stack([u, v, z], dim=-1)

    def transform_points(self, points, eps: Optional[float] = None, **kwargs) -> torch.Tensor:
        screen_points = self.transform_points_screen(points, eps, **kwargs)
        u, v, z = screen_points.unbind(-1)
        
        if self.image_size is not None:
            if self.image_size.shape[0] == 1 or self.image_size.ndim == 1:
                H, W = self.image_size[0] if self.image_size.ndim==2 else self.image_size
            else:
                H, W = self.image_size[0]

            u_ndc = (2.0 * u / (W - 1.0)) - 1.0
            
            v_ndc = 1.0 - (2.0 * v / (H - 1.0))
            
            return torch.stack([u_ndc, v_ndc, z], dim=-1)
        
        return screen_points

    def in_ndc(self):
        return self.image_size is not None

    def is_perspective(self):
        return False
