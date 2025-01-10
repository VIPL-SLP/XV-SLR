# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------

import pdb
import PIL
import copy
import glob
import torch
import random
import pickle
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-4

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton):
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton


class ToTensor(object):
    def __call__(self, skeleton):
        if isinstance(skeleton, np.ndarray):
            skeleton = torch.from_numpy(skeleton).float()
        return skeleton

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, skeleton):
        # B, H, W, 3
        flag = random.random() < self.prob
        if flag:
            skeleton[:, :, 0] = 256 - skeleton[:, :, 0]
            swap_ind = (
                [0, 2, 1, 4, 3, 6, 5, 8, 7]
                # list(range(9))
                + list(range(30, 51))
                + list(range(9, 30))
                # + list(range(51, 77))
                + [55, 54, 53, 52, 51, 58, 57, 56]
                + list(range(59, 76))[::-1]
                + [76]
            )
            skeleton = skeleton[:, swap_ind]
        return skeleton

class RandomResize(object):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, skeleton):
        scale = random.uniform(1 - self.rate, 1 + self.rate)
        skeleton[:, :, :2] = skeleton[:, :, :2] * scale
        return skeleton

class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2) -> None:
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        # clip shape: T X N X 2
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return [clip[i] for i in index] if isinstance(clip, list) else clip[index]


class RandomRot(object):
    def __init__(self, theta=0.3):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, skeleton):
        T, V, C = skeleton.shape

        if np.all(np.isclose(skeleton, 0)):
            return skeleton

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=3)
            rot_mat = self._rot3d(theta)
        elif C == 2:
            theta = np.random.uniform(-self.theta, self.theta)
            # theta = self.theta
            rot_mat = self._rot2d(theta)
        skeleton[:, :, 0] -= 128
        skeleton[:, :, 1] -= 128
        skeleton = np.einsum('ab,tvb->tva', rot_mat, skeleton)
        skeleton[:, :, 0] += 128
        skeleton[:, :, 1] += 128
        return skeleton


class RandomShift(object):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, skeleton):
        x_scale = random.uniform(-self.rate, self.rate)
        y_scale = random.uniform(-self.rate, self.rate)
        z_scale = random.uniform(-self.rate, self.rate)
        skeleton[:, :, 0] += x_scale
        skeleton[:, :, 1] += y_scale
        # skeleton[:, :, 2] += z_scale
        return skeleton


class RandomMask(object):
    def __init__(self, mask_ratio=0.2, splits=[]) -> None:
        self.mask_ratio = mask_ratio
        self.splits = splits

    def __call__(self, clip):
        # clip shape: T X N X 2
        T, N, C = clip.shape
        nparts = len(self.splits) - 1
        mask = np.random.rand(T, nparts) > self.mask_ratio
        for i in range(nparts):
            if i > 0:
                clip[:, self.splits[i] : self.splits[i + 1]][mask[:, i]] = clip[
                    :, self.splits[i] : self.splits[i + 1]
                ][mask[:, i]][:, 0][:, None]
        return clip


class RandomGaussianNoise:
    def __init__(self, sigma=0.01, base='frame', shared=False):
        assert isinstance(sigma, float)
        self.sigma = sigma
        self.base = base
        self.shared = shared
        assert self.base in ['frame', 'video']
        if self.base == 'frame':
            assert not self.shared

    def __call__(self, skeleton):
        # skeleton = results['keypoint']
        N, V, C = skeleton.shape
        # skeleton = skeleton.reshape(-1, V, C)
        ske_min, ske_max = skeleton.min(axis=1), skeleton.max(axis=1)
        # MT * C
        flag = (ske_min ** 2).sum(axis=1) > EPS
        # MT
        if self.base == 'frame':
            norm = np.linalg.norm(ske_max - ske_min, axis=1) * flag
            # MT
        elif self.base == 'video':
            assert np.sum(flag)
            ske_min, ske_max = ske_min[flag].min(axis=0), ske_max[flag].max(axis=0)
            # C
            norm = np.linalg.norm(ske_max - ske_min)
            norm = np.array([norm] * N) * flag
        # MT * V
        if self.shared:
            noise = np.random.randn(V) * self.sigma
            noise = np.stack([noise] * N)
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            random_vec = np.concatenate([random_vec] * N, axis=-1)
        else:
            noise = np.random.randn(N, V) * self.sigma
            noise = (noise.T * norm).T
            random_vec = np.random.uniform(-1, 1, size=(C, N * V))
            random_vec = random_vec / np.linalg.norm(random_vec, axis=0)
            # C * MTV
        random_vec = random_vec * noise.reshape(-1)
        # C * MTV
        random_vec = (random_vec.T).reshape(N, V, C)
        skeleton = skeleton + random_vec
        return skeleton


def view_skeleton_plot(skeleton, save_path):
    plt.scatter(skeleton[:, 0], -skeleton[:, 1], c='blue')
    for i in range(len(skeleton)):
        plt.text(skeleton[i, 0], -skeleton[i, 1], s=str(i))
    plt.xlim(0, 256)
    plt.show()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    pkl_list = sorted(glob.glob('../dataset/phoenix14T_skeleton/fullFrame-256x256px/dev/01April_2010_Thursday_heute-6697/*.pkl'))
    skeleton = np.array([pickle.load(open(pkl_path,'rb',))['keypoints'] for pkl_path in pkl_list])
    kps_idx = [0, 1, 2, 5, 6, 7, 8, 9, 10] + [i - 1 for i in range(92, 113)] + [i - 1 for i in range(113, 134)]
    skeleton = skeleton[:, kps_idx]
    view_skeleton_plot(skeleton[0], "test.jpg")
    aug = RandomResize(0.5)
    skeleton = aug(skeleton)
    view_skeleton_plot(skeleton[0], "test1.jpg")
    # Rotate = RandomRot()
    # view_skeleton(skeleton[0], 'flip2.png')
    # skeleton = Flip(skeleton)
    # skeleton = Rotate(skeleton)
    # skeleton[:,:,0] += 128
    # skeleton[:,:,1] += 128
    # view_skeleton(skeleton[0], 'rotate.png')
    # view_skeleton_plot(skeleton[0])
