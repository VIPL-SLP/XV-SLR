# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------
import os

import pdb
import PIL
import copy
import glob
import torch
import random
import pickle
import matplotlib
import numpy as np
import numbers
from PIL import Image
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.visualize3d import get_rotation,perspective_projection

EPS = 1e-4

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton, fid=None, osxposs=0.5, pose_idx=None):
        for t in self.transforms:
            if isinstance(t,RandomOSX):
                skeleton = t(skeleton, fid, osxposs, pose_idx)
            else:
                skeleton = t(skeleton)
        return skeleton

class ToTensor(object):
    def __call__(self, data):
        for k, v in data.items():
            if k == '2d_skeleton' or k == 'r_features' or k == 'd_features':
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).float()
            elif k == 'rgb' or k == 'depth':
                # transform from T,H,W,C to T,C,H,W
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v.transpose((0, 3, 1, 2))).float()
            else:
                raise ValueError(f'unknown data type {k}')
        return data

class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, data):
        # not apply for skeleton data
        data_type = list(data.keys())
        need_modify = [x for x in data_type if x in ['rgb', 'depth']]
        if len(need_modify) == 0:
            # print("crop shape:",data['2d_skeleton'].shape)
            return data
        crop_h, crop_w = self.size
        _, im_h, im_w, _ = data[need_modify[0]].shape
        for item in need_modify:
            clip = data[item]
            if crop_w > im_w:
                pad = crop_w - im_w
                clip = [np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), 'constant', constant_values=0) for img in
                        clip]
                w1 = 0
            else:
                w1 = random.randint(0, im_w - crop_w)

            if crop_h > im_h:
                pad = crop_h - im_h
                clip = [np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), 'constant', constant_values=0) for img in
                        clip]
                h1 = 0
            else:
                h1 = random.randint(0, im_h - crop_h)
            data[item] = np.array([img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip])
        return data

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, data):
        # same as random crop
        data_type = list(data.keys())
        need_modify = [x for x in data_type if x in ['rgb', 'depth']]
        if len(need_modify) == 0:
            return data
        new_h, new_w = self.size
        _, im_h, im_w, _ = data[need_modify[0]].shape
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.))
        left = int(round((im_w - new_w) / 2.))
        for item in need_modify:
            clip = data[item]
            data[item] = np.array([img[top:top + new_h, left:left + new_w] for img in clip])
        return data

class RandomBGPerturb(object):
    def __init__(self, prob, bg_dir='MM_WLAuslan/train/kf/bg/') -> None:
        self.prob = prob
        bgs = sorted(os.listdir(bg_dir))
        assert len(bgs) > 0
        self.bgs = [Image.open(os.path.join(bg_dir, item)) for item in bgs]
        print(f'apply bg bg numbers: {len(self.bgs)}')
        for i in range(len(self.bgs)):
            self.bgs[i].load()
    
    def __call__(self, data):
        # rgb and mask must be provided for this augmentation
        if (not 'rgb' in data.keys()) or (not 'mask' in data.keys()):
            return data
        assert len(data['rgb']) == len(data['mask'])
        flag = random.random() < self.prob
        if flag:
            bg = random.choice(self.bgs)
            for i in range(len(data['rgb'])):
                source_image = data['rgb'][i]
                mask = data['mask'][i]
                data['rgb'][i] = Image.composite(data['rgb'][i], bg.resize(source_image.size), mask.resize(source_image.size, Image.LANCZOS))
        data['rgb'] = [np.array(item.convert('RGB')) for item in data['rgb']]
        data['rgb'] = np.stack(data['rgb'], axis=0)
        del data['mask'] # useless, del it
        return data

class RandomHorizontalFlip(object):
    def __init__(self, prob, skeleton_width=512):
        self.prob = prob
        self.skeleton_width = skeleton_width
    
    def flip_skeleton(self, skeleton):
        # skeleton shape: T,N,C
        skeleton[:, :, 0] = self.skeleton_width - skeleton[:, :, 0]
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

    def flip_rgb(self, clip):
        # clip shape: T,H,W,C C can be 1 for depth, 3 for rgb or 4 for rgb+depth
        clip = np.flip(clip, axis=2)
        clip = np.ascontiguousarray(copy.deepcopy(clip))
        return np.array(clip)

    def __call__(self, data):
        # B, H, W, 3
        flag = random.random() < self.prob
        if flag:
            for k, v in data.items():
                if k == '2d_skeleton':
                    data[k] == self.flip_skeleton(v)
                elif k in ['rgb', 'depth']:
                    data[k] = self.flip_rgb(v)
                else:
                    raise ValueError(f'unknown data type {k}')
        return data

class RandomResize(object):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, data):
        # not apply for rgb data
        data_type = list(data.keys())
        need_modify = [x for x in data_type if x in ['2d_skeleton']]
        scale = random.uniform(1 - self.rate, 1 + self.rate)
        if len(need_modify) > 0:
            for k in need_modify:
                data[k][:, :, :2] = data[k][:, :, :2] * scale
        return data

class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2) -> None:
        self.min_len = 32 # 16
        self.max_len = 230 # 380
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, data):
        data_type = list(data.keys())
        vid_len = len(data[data_type[0]])
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
        for k, v in data.items():
            data[k] = v[index]
        return data

class TemporalFixedRescale(object):
    def __init__(self, fixed_orig_poss, dec_poss) -> None:
        self.min_len = 32 # 16
        self.max_len = 230 # 380
        self.fixed_orig_poss = fixed_orig_poss
        self.dec_poss = dec_poss

    def __call__(self, data):
        poss = random.random()
        if poss < self.fixed_orig_poss:
            return data
        poss = random.random()
        if poss < self.dec_poss:
            ratio = 0.5
        else: ratio = 2
        data_type = list(data.keys())
        vid_len = len(data[data_type[0]])
        new_len = int(vid_len * ratio)
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            # index = sorted([i for i in range(0,vid_len,vid_len//new_len)])
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            # orig_lst = [i for i in range(vid_len)]
            # index = sorted(np.linspace(orig_lst[0],orig_lst[-1],new_len).tolist())
            # index = [int(i) for i in index]
            # print("orig:",orig_lst)
            # print("new:",index) # used to check the correctness
            index = sorted(random.choices(range(vid_len), k=new_len))
        for k, v in data.items():
            data[k] = v[index]
        return data

class TemporalRemove(object):
    def __init__(self,remove_orig_poss, rm_front_poss) -> None:
        self.min_len = 32 # 16
        self.max_len = 230 # 380
        self.remove_orig_poss = remove_orig_poss
        self.rm_front_poss = rm_front_poss

    def __call__(self, data):
        poss = random.random()
        if poss < self.remove_orig_poss:
            return data
        data_type = list(data.keys())
        vid_len = len(data[data_type[0]])
        remove_num = random.randint(1,vid_len//4)
        new_len = vid_len - remove_num
        
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len < vid_len:
            poss = random.random()
            if poss < self.rm_front_poss:
                index = [i for i in range(vid_len - new_len,vid_len)]
            else: index = [i for i in range(0, new_len)]
            for k, v in data.items():
                data[k] = v[index]
        return data

class RandomRot(object):
    def __init__(self, theta=0.3, skeleton_shape=(256, 256)):
        self.theta = theta
        self.skeleton_shape = skeleton_shape

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

    def __call__(self, data):
        # not apply for rgb data
        if '2d_skeleton' in data.keys():
            # assert we will put the conf score in the last dim
            skeleton, conf = data['2d_skeleton'][:,:,:-1], data['2d_skeleton'][:,:,-1:]
            T, V, C = skeleton.shape

            if np.all(np.isclose(skeleton, 0)):
                return skeleton

            assert C in [2, 3]
            if C == 3:
                theta = np.random.uniform(-self.theta, self.theta, size=3)
                rot_mat = self._rot3d(theta)
            elif C == 2:
                theta = np.random.uniform(-self.theta, self.theta)
                rot_mat = self._rot2d(theta)
            for i in range(C):
                skeleton[:, :, i] -= (self.skeleton_shape[i] // 2)
            skeleton = np.einsum('ab,tvb->tva', rot_mat, skeleton)
            for i in range(C):
                skeleton[:, :, i] += (self.skeleton_shape[i] // 2)
            data['2d_skeleton'] = np.concatenate([skeleton, conf], axis=-1)
        return data

class RandomOSX(object):
    def __init__(self,fid,osxposs, pose_idx):
        self.osxposs = osxposs
        self.pose_idx = pose_idx

    def __call__(self, data, fid,osxposs, pose_idx):
        if '2d_skeleton' not in data.keys():
            return data

        poss = random.random()
        if poss > self.osxposs:
            return data
        pkl_file = f'./MM_WLAuslan/train/kf/2d_skeleton/{fid}_kf_rgb_kps3d.npy'
        # 3D骨架点存储方式是个字典
        assert(os.path.exists(pkl_file))
        # pdb.set_trace()
        info = np.load(pkl_file, allow_pickle=True).item()
        if 'skeleton' not in info.keys():
            return data
        # print("keys:",info.keys())
        # angle = [-10,30,0]
        angle = [random.randint(-10,10),random.randint(-30,30),0]
        
        global_rot = torch.from_numpy(
                get_rotation(
                    # For x-axis, [-10, 10] are suitable
                    theta_x=np.radians(angle[0]), 
                    # For y-axis, [-30, 30] are suitable
                    theta_y=np.radians(angle[1]),
                    # For z-axis, there is no need to adjust
                    theta_z=np.radians(angle[2]),
                    )
                ).unsqueeze(0).expand(len(info['skeleton']), -1, -1).float()
        # T * 133 * 2
        projected_2d = perspective_projection(
                info['skeleton'], 
                global_rot,
                info['global_translation'],
                info['focal'],
                info['princpt'],
            )
        # 要抽出body和left/right hand!!!
        skeleton_data = (projected_2d.cpu().detach().numpy())
        skeleton_data = skeleton_data[:,self.pose_idx] # [T,51,2]
        # print("osx pose_idx:",self.pose_idx)
        skeleton_data = np.concatenate((skeleton_data,np.ones((skeleton_data.shape[0],skeleton_data.shape[1], 1))),axis=-1)
        # print("osx shape:",skeleton_data.shape)
        data['2d_skeleton'] = skeleton_data
        return data

class RandomShift(object):
    # useless if we use cosign normalization, leave it alone
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
    # if you want to use random mask, please put it before random crop as this operation will
    # change the size of frames and make it unmatch the coordinates of skeleton
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

    def __call__(self, data):
        # not apply for rgb
        if not '2d_skeleton' in data.keys():
            return data
        skeleton, conf = data['2d_skeleton'][:,:,:-1], data['2d_skeleton'][:,:,-1:]
        N, V, C = skeleton.shape
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

        data['2d_skeleton'] = np.concatenate([skeleton, conf], axis=-1)

        return data

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
