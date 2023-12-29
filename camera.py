from typing import Dict, Tuple
import torch
import cv2
import kornia
from kornia.geometry.liegroup.se3 import Se3, So3
from utils import ssd_loss, warp, image_gradient

class Camera:
    def __init__(self, config: Dict) -> None:
        self._width = config["width"]
        self._height = config["height"]
        self.fx = config["intrinsicMatrix"][0][0]
        self.fy = config["intrinsicMatrix"][1][1]
        self.cx = config["intrinsicMatrix"][0][2]
        self.cy = config["intrinsicMatrix"][1][2]
        self._K = torch.Tensor(config["intrinsicMatrix"])
        R = torch.Tensor(config["extrinsics"]["rotationMatrix"])
        _t = config["extrinsics"]["translation"]
        T = torch.Tensor([_t["x"], _t["y"], _t["z"]])
        self._pose = Se3(So3.from_matrix(R), T)

    def __init__(self, width, height, fx, fy, cx, cy) -> None:
        self._width = width
        self._height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self._K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self._pose = Se3.identity()

    @property
    def K(self) -> torch.Tensor:
        return self._K

    @property
    def K_inv(self):
        return self._K.inverse()
    
    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def pose(self) -> Se3:
        return self._pose

    @pose.setter
    def pose(self, pose: Se3) -> None:
        self._pose = pose

    def uv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.meshgrid(torch.arange(0, self._height), torch.arange(0, self._width))

    def project(self, points: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.K, points.reshape(3, -1)).reshape(-1, 3)

    def projecti(self, points: torch.Tensor):
        points[:, :3] = self.project(points[:, :3])
        mask = (
            (points[:, 0] >= 0)
            & (points[:, 0] < self._height)
            & (points[:, 1] >= 0)
            & (points[:, 1] < self._width)
        )
        points = points[mask]
        img = torch.zeros((self._height, self._width), device=points.device)
        x_indices = points[:, 0].to(torch.int64)
        y_indices = points[:, 1].to(torch.int64)
        img[x_indices, y_indices] = points[:, 3]
        return img

    def unproject(self) -> torch.Tensor:
        u, v = self.uv()
        uv = torch.stack([u, v, torch.ones([self._height, self._width])], -1)
        return torch.matmul(self.K_inv, uv.reshape(3, -1)).reshape(-1, 3)

    def unprojecti(self, img) -> torch.Tensor:
        u, v = self.uv()
        uv = torch.stack([u, v, torch.ones([self._height, self._width])], -1)
        pts = torch.matmul(self.K_inv, uv.reshape(3, -1)).reshape(-1, 3)
        return torch.concat((pts, img.flatten().unsqueeze(-1)), 1)


# class OAKD:
#     left: Camera
#     right: Camera
#     left_data: cv2.VideoCapture
#     righ_data: cv2.VideoCapture

#     def __init__(
#         self, config: Dict, left_data: cv2.VideoCapture, right_data: cv2.VideoCapture
#     ):
#         self.left_data = left_data
#         self.right_data = right_data

#         self.left = Camera(config["cameraData"][1][1])
#         self.right = Camera(config["cameraData"][2][1])

#     def read(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         loop = True
#         while (loop):
#             lret, left_frame = self.left_data.read()
#             rret, right_frame = self.right_data.read()
#             loop = lret & rret
#             if not loop:
#                 break
#             yield kornia.image_to_tensor(left_frame), kornia.image_to_tensor(right_frame)

#     def project(self):
#         pass

#     def unproject(self):
#         pass

#     def warp(points: torch.Tensor, pose: Se3):
#         w = img.shape[2]
#         h = img.shape[1]
#         u, v = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
