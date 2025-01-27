# ----------------------------------------
# Written by Yuecong Min
# ----------------------------------------
import os
import copy
import torch
import random
import numpy as np
import numbers
from PIL import Image
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
        swap_ind = ( # only use the efficient keypoints
            [0, 2, 1, 4, 3, 6, 5, 8, 7]
            + list(range(30, 51))
            + list(range(9, 30))
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
        # 3D keypoints are stored in dict format
        assert(os.path.exists(pkl_file))
        info = np.load(pkl_file, allow_pickle=True).item()
        if 'skeleton' not in info.keys():
            return data
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
        # extract kps of body and left/right hand
        skeleton_data = (projected_2d.cpu().detach().numpy())
        skeleton_data = skeleton_data[:,self.pose_idx] # [T,51,2]
        skeleton_data = np.concatenate((skeleton_data,np.ones((skeleton_data.shape[0],skeleton_data.shape[1], 1))),axis=-1)
        data['2d_skeleton'] = skeleton_data
        return data