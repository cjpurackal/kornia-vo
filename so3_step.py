import torch
import cv2
from utils import image_gradient
from kornia.geometry.liegroup.so3 import So3


def warp(img : torch.Tensor, so3):
    _, _, rows, cols = img.shape
    M = so3.matrix().detach().numpy()
    return torch.from_numpy(cv2.warpPerspective(img[0][0].numpy(), M, (cols, rows))).reshape(1, 1, rows, cols).float()

def get_jac_row(grey_ref_frame, cam_rot, height, width, K):
    grey_ref_warped = warp(
        grey_ref_frame,
        cam_rot,
    )
    wI_grad_x, wI_grad_y = image_gradient(grey_ref_warped)
    I_grad_x, I_grad_y = image_gradient(grey_ref_frame)
    I_grad_x = ((wI_grad_x +  I_grad_x) / 2)[0][0]
    I_grad_y = ((wI_grad_y + I_grad_y) / 2)[0][0]

    y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    z = torch.ones_like(x)
    unwraped_point = torch.stack((x, y, torch.ones_like(x)), -1).float()
    point = K.inverse().view(1, 3, 3) @ unwraped_point.view(-1, 3, 1)
    K_Rlr = (K @ cam_rot.matrix())
    a, b, c= K_Rlr[0, 0], K_Rlr[0, 1], K_Rlr[0, 2]
    d, e, f = K_Rlr[1, 0], K_Rlr[1, 1], K_Rlr[1, 2]
    g, h, i = K_Rlr[2, 0], K_Rlr[2, 1], K_Rlr[2, 2]

    ele1 = (z * (d * I_grad_y + a * I_grad_x) - (I_grad_y * g * y) - (I_grad_x * g * x)) / (z ** 2)
    ele2 = (z * (e * I_grad_y + b * I_grad_x) - (I_grad_y * h * y) - (I_grad_x * h * x)) / (z ** 2)
    ele3 = (z * (f * I_grad_y + c * I_grad_x) - (I_grad_y * i * y) - (I_grad_x * i * x)) / (z ** 2)
    leftproduct = torch.stack((ele1.reshape(-1), ele2.reshape(-1), ele3.reshape(-1)), -1)
    return leftproduct.cross(point.view(-1, 3))


# def error(grey_ref_frame, grey_cur_frame, cam_rot):
#     jac_row = get_jac_row()
#     residual = warp(grey_ref_frame, cam_rot) - grey_cur_frame
#     B = residual.view(-1, 1) * jac_row
#     B = B.sum(0)
#     return B.view(-1, 3)


def so3_step(grey_ref_frame, grey_cur_frame, cam, rot, iter: int = 10):
    for _ in range(iter):
        jac_row = get_jac_row(grey_ref_frame, rot, cam.height, cam.width, cam.K)
        a, b, c = jac_row[:, 0], jac_row[:, 1], jac_row[:, 2]
        A_row1 = torch.stack((a.pow(2), a * b, a * c), -1)
        A_row2 = torch.stack((a * b, b.pow(2), b * c), -1)
        A_row3 = torch.stack((a * c, b * c, c.pow(2)), -1)
        A = torch.stack((A_row1, A_row2, A_row3), -1)
        A = A.sum(0)
        residual = warp(grey_ref_frame, rot) - grey_cur_frame
        B = residual.view(-1, 1) * jac_row
        B = B.sum(0)
        LD, pivots, info = torch.linalg.ldl_factor_ex(A)
        delta = torch.linalg.ldl_solve(LD, pivots, B.unsqueeze(-1))
        # TODO This does not equate to zero
        # torch.linalg.norm(A @ delta - B)
        rot_update = So3.exp(delta.squeeze())
        rot = rot * rot_update
    return rot
