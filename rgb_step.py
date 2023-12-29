import torch
from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.liegroup.se3 import Se3
from utils import image_derivative_gsobel, Correspondence


def get_jac_row(cimg: Correspondence, sigma: torch.Tensor, pc: torch.Tensor, di_dx: torch.Tensor, di_dy: torch.Tensor, cam):
    sobelScale = 1.0 / pow(2.0, 3)
    #TO DO: Fix naming
    ww = sigma + cimg.diff
    ww_mask = ww > 1.1920929e-07
    ww[ww_mask] = 1. / ww[ww_mask]
    ww[~ww_mask] = 1.
    row = torch.zeros((cam.height, cam.width, 7))
    row[cimg.u0, cimg.v0, 6] = -ww * cimg.diff

    pc = pc[:, cimg.u1, cimg.v1]
    invz = 1 / pc[2, :]
    di_dx_val = ww * sobelScale * di_dx[0, cimg.u0, cimg.v0]
    di_dy_val = ww * sobelScale * di_dy[0, cimg.u0, cimg.v0]
    v0 = di_dx_val * cam.fx * invz
    v1 = di_dy_val * cam.fy * invz
    v2 = -(v0 * pc[0, :] + v1 * pc[1, :]) * invz

    row[cimg.u0, cimg.v0, 0] = v0
    row[cimg.u0, cimg.v0, 1] = v1
    row[cimg.u0, cimg.v0, 2] = v2
    row[cimg.u0, cimg.v0, 3] = -pc[2, :] * v1 + pc[1, :] * v2
    row[cimg.u0, cimg.v0, 4] = pc[2, :] * v0 - pc[0, :] * v2
    row[cimg.u0, cimg.v0, 5] = -pc[1, :] * v0 + pc[0, :] * v1

    row_0, row_1, row_2, row_3, row_4, row_5, row_6 = row[:, :, 0].flatten(), row[:, :, 1].flatten(), row[:, :, 2].flatten(), row[:, :, 3].flatten(), row[:, :, 4].flatten(), row[:, :, 5].flatten(), row[:, :, 6].flatten()
    values = torch.stack([
        row_0 * row_0, row_0 * row_1, row_0 * row_2, row_0 * row_3, row_0 * row_4, row_0 * row_5, row_0 * row_6,
        row_1 * row_1, row_1 * row_2, row_1 * row_3, row_1 * row_4, row_1 * row_5, row_1 * row_6,
        row_2 * row_2, row_2 * row_3, row_2 * row_4, row_2 * row_5, row_2 * row_6,
        row_3 * row_3, row_3 * row_4, row_3 * row_5, row_3 * row_6,
        row_4 * row_4, row_4 * row_5, row_4 * row_6,
        row_5 * row_5, row_5 * row_6,
        row_6 * row_6
    ])
    values = values.sum(1)
    return values

def rgb_step(cimg: Correspondence, sigma: torch.Tensor, pc: torch.Tensor, di_dx: torch.Tensor, di_dy: torch.Tensor, cam):
    values = get_jac_row(cimg, sigma, pc, di_dx, di_dy, cam)
    shift = 0
    RT = torch.zeros((4, 4))
    A = torch.zeros((6, 6))
    B = torch.zeros((6, 1))
    for i in range(6):
        for j in range(i, 7):
            val = values[shift]
            shift += 1
            if j == 6:
                 B[i] = val
            else:
                A[i, j] = val
                A[j, i] = val
    LD, pivots, _ = torch.linalg.ldl_factor_ex(A)
    delta = torch.linalg.ldl_solve(LD, pivots, B)
    rot = So3.exp(delta[:3].squeeze())
    trans = delta[3:]
    RT[:3, :3] = rot.matrix()
    RT[:3, 3] = trans.squeeze()
    RT[3, 3] = 1
    return RT

