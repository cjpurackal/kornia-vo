import torch
import torch.nn.functional as F
import theseus as th
from kornia.filters import spatial_gradient
from kornia.geometry.liegroup.so3 import So3


class RGBD:
    rgb: torch.Tensor
    depth: torch.Tensor
    def __init__(self, rgb: torch.Tensor, depth: torch.Tensor) -> None:
        self.rgb = rgb
        self.depth = depth


def compute_trifocal_tensor(cam, ref_pose_curr):
    T = (
        cam.right.K
        * cam.right.pose.so3.matrix()
        * cam.left.K.inverse()
        * cam.left.K
        * ref_pose_curr.t
        * cam.left.pose.t
    ) - (
        cam.right.K
        * cam.right.pose.t
        * cam.left.K
        * ref_pose_curr.so3.matrix()
        * cam.left.pose.t
        * cam.left.K.inverse()
    )
    return T


def ssd_loss(x: torch.Tensor, y: torch.Tensor):
    return torch.sum(torch.square(x - y))


def image_gradient(img: torch.Tensor) -> torch.Tensor:
    x = F.pad(img, (1, 1, 1, 1), mode="reflect")
    dx2 = x[..., 1:-1, :-2] - x[..., 1:-1, 2:]
    dy2 = x[..., :-2, 1:-1] - x[..., 2:, 1:-1]
    return dx2, dy2

def image_derivative_gsobel(img: torch.Tensor, gsobel_x3x3: torch.Tensor = None, gsobel_y3x3: torch.Tensor = None):
    def apply_sobel(x, gsobel):
        img_0 = x[..., 1:-1, :-2] * gsobel[8] + x[..., 1:-1, 1:-1] * gsobel[7] + x[..., 1:-1, 2:] * gsobel[6]
        img_1 = x[..., 1:-1, :-2] * gsobel[5] + x[..., 1:-1, 1:-1] * gsobel[4] + x[..., 1:-1, 2:] * gsobel[3]
        img_2 = x[..., 1:-1, :-2] * gsobel[2] + x[..., 1:-1, 1:-1] * gsobel[1] + x[..., 1:-1, 2:] * gsobel[0]
        img_0 = F.pad(img_0, (1, 1, 1, 1), mode="reflect")
        img_1 = F.pad(img_1, (1, 1, 1, 1), mode="reflect")
        img_2 = F.pad(img_2, (1, 1, 1, 1), mode="reflect")
        return img_0[..., :-2, 1:-1] + img_1[..., 1:-1, 1:-1] + img_2[..., 2:, 1:-1]

    gsobel_x3x3 = torch.tensor([0.52201, 0.00000, -0.52201, 0.79451, -0.00000, -0.79451, 0.52201, 0.00000, -0.52201])
    gsobel_y3x3 = torch.Tensor([0.52201, 0.79451, 0.52201, 0.00000, 0.00000, 0.00000, -0.52201, -0.79451, -0.52201])
    x = F.pad(img.float(), (1, 1, 1, 1), mode="reflect")
    dx = apply_sobel(x, gsobel_x3x3)
    dy = apply_sobel(x, gsobel_y3x3)
    return dx, dy


def warp(grey: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, T, grey_lookup: torch.Tensor = None) -> torch.Tensor:
    if grey_lookup is None:
        grey_lookup = grey
    height = grey.shape[2]
    width = grey.shape[3]
    #get the current frame pixel indices
    u, v = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    u = u / height
    v = v / width
    #add depth to the pixel indices
    uvz = torch.cat((u.flatten().unsqueeze(-1), v.flatten().unsqueeze(-1), torch.ones(height*width).unsqueeze(-1)), -1)
    uvz = torch.where(depth.flatten().unsqueeze(-1) != 0, uvz * depth.flatten().unsqueeze(-1), uvz)
    #unproject the pixel indices
    xyz = torch.matmul(K.inverse(), uvz.reshape(3, -1)).reshape(-1, 3)
    #transform the pixel indices to the current frame
    xyz = T.transform_from(xyz).tensor
    #project the pixel indices to the current frame
    uvz_ = torch.matmul(K, xyz.reshape(3, -1)).reshape(-1, 3)
    #normalize the pixel indices
    uvz_ =  uvz_ / uvz_[:, 2].unsqueeze(-1)
    u_ = uvz_[:, 0].reshape(height, width)
    v_ = uvz_[:, 1].reshape(height, width) 
    #convert to -1, 1 range
    u_ = 2 * u_ - 1
    v_ = 2 * v_ - 1
    #get the current frame pixel values
    warped_frame = torch.nn.functional.grid_sample(grey_lookup, torch.stack((v_, u_,), dim=-1).unsqueeze(0))
    return warped_frame


def get_valid_neighbor_mask(image: torch.Tensor):
        image = image.squeeze()    
        left = F.pad(image, (1, 1), "constant", 1)[:,:-2]
        right = F.pad(image, (1, 1), "constant", 1)[:,2:]
        top = F.pad(image, (0, 0, 1, 1), "constant", 1)[:-2, :]
        bottom = F.pad(image, (0, 0, 1, 1), "constant", 1)[2:, :]
        top_left = F.pad(image, (1, 1, 1, 1), "constant", 1)[:-2, :-2]
        top_right = F.pad(image, (1, 1, 1, 1), "constant", 1)[:-2, 2:]
        bottom_left = F.pad(image, (1, 1, 1, 1), "constant", 1)[2:, :-2]
        bottom_right = F.pad(image, (1, 1, 1, 1), "constant", 1)[2:, 2:]
        mask = (image > 0) & (left > 0) & (right > 0) & (top > 0) & (bottom > 0) & (top_left > 0) & (top_right > 0) & (bottom_left > 0) & (bottom_right > 0) 
        return mask

def get_grad_mask(image: torch.Tensor):
        minimumGradientMagnitudes = 5
        sobelScale = 1.0 / pow(2.0, 3)
        min_scale = pow(minimumGradientMagnitudes, 2.0) / pow(sobelScale, 2.0)
        dimg = spatial_gradient(image.float()).squeeze()
        di_dx, di_dy = dimg[0], dimg[1]
        m_two = di_dx * di_dx + di_dy * di_dy
        mask = (m_two >= min_scale)
        return mask

class Correspondence:
    def __init__(self, last_image: torch.Tensor, next_image: torch.Tensor, last_depth: torch.Tensor, next_depth: torch.Tensor, rot: So3, trans: torch.Tensor, K: torch.Tensor):
        maxDepthDelta = 5
        height = last_image.shape[2]
        width = last_image.shape[3]
        uv = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))

        krkinv = K @ rot.matrix() @ K.inverse()
        kt = K @ trans

        valid_mask = get_valid_neighbor_mask(last_image)
        grad_mask = get_grad_mask(last_image)
        next_depth = next_depth.squeeze()
        next_depth_mask = next_depth > 0

        mask = valid_mask & grad_mask & next_depth_mask
        u1, v1 = uv[0][mask], uv[1][mask]
        next_depth = next_depth[mask]

        #TODO: Vectorize this
        trans_d1 = next_depth * (krkinv[2, 0] * u1 + krkinv[2, 1] * v1 + krkinv[2, 2]) + kt[2]
        u0 = next_depth * ((krkinv[0, 0] * u1 + krkinv[0, 1] * v1 + krkinv[0, 2]) + kt[0]) // trans_d1
        v0 = next_depth * ((krkinv[1, 0] * u1 + krkinv[1, 1] * v1 + krkinv[1, 2]) + kt[1]) // trans_d1

        u0_mask = (u0 >= 0) & (u0 < height)
        v0_mask = (v0 >= 0) & (v0 < width)
        mask = u0_mask & v0_mask
        u0, v0 = u0[mask].long(), v0[mask].long()
        u1, v1 = u1[mask].long(), v1[mask].long()
        last_depth_mask = last_depth[0, 0, u0, v0] > 0
        last_depth = last_depth[0, 0, u0, v0][last_depth_mask]

        #TO DO: depth_delta_mask = trans_d1 - last_depth < maxDepthDelta
        diff = next_image[0, 0, u1, v1] - last_image[0, 0, u0, v0]
        
        self.u0 = u0
        self.v0 = v0
        self.u1 = u1
        self.v1 = v1
        self.diff = diff


def get_point_cloud(depth: torch.Tensor, cx: float, cy: float, fx: float, fy: float):
    height = depth.shape[2]
    width = depth.shape[3]
    u, v = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    x = (v - cx) * depth / fx
    y = (u - cy) * depth / fy
    #TO DO: Fix the shape
    return torch.cat((x, y, depth), 1)[0]