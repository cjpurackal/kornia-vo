import torch
import cv2
import kornia

class Dysonlab():
    def __init__(self, rgb_files, depth_files=None, gt_poses_file=None) -> None:
        with open(rgb_files) as f:
            self.rgb_files = [line.split("\n")[0] for line in list(f)]
        if depth_files:
            with open(depth_files) as f:
                self.depth_files = [line.split("\n")[0] for line in list(f)]
        
    def read_grey(self, scale=1, normalize=False) -> torch.Tensor:
        for f in self.rgb_files:
            frame = cv2.imread(f, 0)
            frame = cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale), interpolation = cv2.INTER_LINEAR)
            if normalize:
                frame = cv2.normalize(frame, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            yield kornia.image_to_tensor(frame, keepdim=True)

    def read_depth(self, scale=1) -> torch.Tensor:
        for f in self.depth_files:
            frame = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            frame = cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale), interpolation = cv2.INTER_LINEAR)
            frame = cv2.normalize(frame, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            yield kornia.image_to_tensor(frame, keepdim=True)

if __name__ == "__main__":
    loader = Dysonlab("rgb.txt")