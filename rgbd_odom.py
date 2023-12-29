import torch
from kornia.geometry.liegroup.so3 import So3
from kornia.geometry.liegroup.se3 import Se3
import rerun as rr
from data.dysonlab.loader import Dysonlab
from so3_step import so3_step
from rgb_step import rgb_step
from camera import Camera
from utils import Correspondence, get_point_cloud, image_derivative_gsobel


rr.init("my_app", spawn = True)
width = 640
height = 480
fx = 528
fy = 528
cx = 320
cy = 240

cam = Camera(width, height, fx, fy, cx, cy)
loader = Dysonlab(
    "data/dysonlab/rgb.txt",
    "data/dysonlab/depth.txt",
)
grey_ref_frame = next(loader.read_grey()).unsqueeze(0).float()
depth_ref_frame = next(loader.read_depth()).unsqueeze(0).float()

grey_cur_frame = next(loader.read_grey()).unsqueeze(0).float()
depth_cur_frame = next(loader.read_depth()).unsqueeze(0).float()

trans = torch.tensor([0., 0., 0.])
pose = Se3.identity()

RT = torch.zeros((4, 4))
RT_prev = torch.zeros((4, 4))
rot = So3.identity()
so3_rot = so3_step(grey_ref_frame, grey_cur_frame, cam, rot)
RT[:3, :3] = so3_rot.matrix()


for grey_cur_frame, depth_cur_frame, in zip(loader.read_grey(), loader.read_depth()):
    for _ in range(10):
        trans = RT[:3, 3]
        cimg = Correspondence(grey_ref_frame, grey_cur_frame[None], depth_ref_frame, depth_cur_frame[None], rot, trans, cam.K)
        sigma = (cimg.diff * cimg.diff).sum()
        pc = get_point_cloud(depth_cur_frame[None], cam.cx, cam.cy, cam.fx, cam.fy)
        di_dx, di_dy = image_derivative_gsobel(grey_cur_frame)
        rgb_RT = rgb_step(cimg, sigma, pc, di_dx, di_dy, cam)
        #update pose
        RT = RT @ rgb_RT
    # cur_frame_3c, ref_frame_3c = torch.ones((480, 640, 3)), torch.ones((480, 640, 3))
    # cur_frame_3c[:, :, 0], ref_frame_3c[:, :, 0] = grey_cur_frame[0], grey_ref_frame[0][0]
    # cur_frame_3c[cimg.u1, cimg.v1, :], ref_frame_3c[cimg.u0, cimg.v0, :] = torch.Tensor([0, 255, 0]), torch.Tensor([0, 255, 0])
    # rr.log_image("cur_frame", cur_frame_3c)
    rr.log_image("cur_frame", grey_cur_frame)
    # rr.log_points("world/point_cloud", pc)
    # check if RT is nan, if so, reset to RT_prev
    if torch.isnan(RT).any():
        RT = RT_prev
    trans_ = RT[:3, 3]
    rot_ = So3.from_matrix(RT[:3, :3])
    rr.log_rigid3("world/camera/#0", parent_from_child=(trans_.tolist(), rot_.q.data.tolist()))
    rr.log_pinhole("world/camera/#0/image", child_from_parent=cam.K, width=cam.width, height=cam.height)
    RT_prev = RT
    grey_ref_frame = grey_cur_frame[None]
    depth_ref_frame = depth_cur_frame[None]