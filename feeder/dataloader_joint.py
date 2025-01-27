import os

import sys
import json
import glob
import yaml
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from utils import joint_augmentation
sys.path.append("..")
stride = 16
class JointFeeder(data.Dataset):
    def __init__(
            self, gloss_dict, input_list_file, num_frames, data_type=['rgb', '2d_skeleton', 'depth','r_features','d_features'], 
            mode="train", transform_mode=True, kps_config=None, view=None,
            osxposs=0.3,temporaltype=0,
            ):
        self.mode = mode
        self.view = view
        self.dict = gloss_dict
        self.data_type = data_type
        self.num_frames = num_frames
        if 'r_features' in self.data_type or 'd_features' in self.data_type:
            self.num_frames = num_frames // stride
        self.inputs_list = json.load(open(input_list_file, 'r'))
        self.transform_mode = "train" if transform_mode else "test"
        self.temporaltype = temporaltype
        print(mode, len(self))
        if kps_config is not None:
            with open(kps_config, 'r') as f:
                self.kps_info = yaml.load(f, Loader=yaml.FullLoader)
        self.pose_idx = sum([v['kps_index'] for k, v in self.kps_info.items()], [])
        self.module_keys = [v['module_key'] for k, v in self.kps_info.items()]
        self.osxposs = osxposs
        self.spatial_data_aug = self.spatial_transform(fid=None,osxposs=self.osxposs)
        self.temporal_data_aug = self.temporal_transform()

    def __getitem__(self, idx):
        # load file info
        fi = self.inputs_list[idx]
        fid, gloss, save_path, view = fi['fid'], fi['label'], fi['save_path'], fi['view']
        if self.view is not None:
            # if the view is set, all samples will load the specific view
            view = self.view
            fi['view'] = view
        label = self.dict['gloss2id'][gloss]
        data = self.read_data(fid, save_path, view)
        data = self.spatial_data_aug(data,fid,self.osxposs)
        data = self.temporal_data_aug(data)
        data = self.normalize(data)
        data['label'] = int(label)
        data['info'] = fi
        return data
    
    def _load_skeleton(self, sks_path):
        pose = np.load(sks_path)
        return pose[:, self.pose_idx]

    def _load_video(self, video_dir, dtype='jpg'):
        all_imgs = sorted(glob.glob(f"{video_dir}/*.{dtype}"))
        video = [np.array(Image.open(img_path).convert('RGB')) for img_path in all_imgs]
        return np.stack(video, axis=0) # T,H,W,3

    def _load_video_PIL(self, video_dir, dtype='jpg'):
        # mask should keep in PIL Image and do not convert to RGB
        all_imgs = sorted(glob.glob(f"{video_dir}/*.{dtype}"))
        video = [Image.open(img_path) for img_path in all_imgs]
        return video

    def _load_features(self, ft_path):
        ft = np.load(ft_path)
        return ft
    
    def read_data(self, fid, path, view):
        data = {}
        for data_type in self.data_type:
            if data_type == '2d_skeleton':
                data[data_type] = self._load_skeleton(os.path.join(path, view, data_type, f'{fid}_{view}_rgb.npy'))
            elif data_type == 'r_features':
                data[data_type] = self._load_features(os.path.join(path.replace('MM_WLAuslan', 'RGBFEAT'), view, data_type, f'{fid}.npy'))
            elif data_type == 'd_features':
                data[data_type] = self._load_features(os.path.join(path.replace('MM_WLAuslan', 'DEPTHFEAT'), view, data_type, f'{fid}.npy'))
            else:
                if 'mask' in self.data_type:
                    # if you load mask, make sure the bgperturb augmentation is used
                    if data_type in ['mask', 'rgb']:
                        data[data_type] = self._load_video_PIL(os.path.join(path, view, data_type, str(fid)))
                    else:
                        data[data_type] = self._load_video(os.path.join(path, view, data_type, str(fid)))
                else:
                    data[data_type] = self._load_video(os.path.join(path, view, data_type, str(fid)))
        return data

    def sample_clip(self, data):
        all_keys = list(data.keys())
        lgt = len(data[all_keys[0]])
        if self.temporaltype == 'LSTM':
            return data
        if self.transform_mode == "train":
            if lgt >= self.num_frames:
                start_frame = np.random.randint(0, max(lgt-self.num_frames, 1))
                clip_idx = [i for i in range(start_frame, start_frame + self.num_frames)]
            else:
                res_frames = self.num_frames - lgt
                lpad = np.random.randint(0, max(res_frames, 1))
                rpad = res_frames - lpad
                clip_idx = [0] * lpad + [i for i in range(lgt)] + [lgt-1] * rpad
        else: # test
            if lgt >= self.num_frames:
                start_frame = (lgt-self.num_frames) // 2
                clip_idx = [i for i in range(start_frame, start_frame + self.num_frames)]
            else:
                res_frames = self.num_frames - lgt # 64 - x
                lpad = res_frames // 2 # 32 - x/2
                rpad = res_frames - lpad # 32 - x/2 
                clip_idx = [0] * lpad + [i for i in range(lgt)] + [lgt-1] * rpad
        for k, v in data.items():
            if k == 'r_features' or k == 'd_features':
                continue
            data[k] = v[clip_idx]
        return data

    def spatial_transform(self,fid,osxposs):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return joint_augmentation.Compose([
                joint_augmentation.RandomOSX(fid,osxposs,self.pose_idx),
            ])
        else:
            print("Apply testing transform.")
            return joint_augmentation.Compose([
                joint_augmentation.CenterCrop(224),
            ])
    
    def temporal_transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return joint_augmentation.Compose([
                joint_augmentation.ToTensor(),
            ])
        else:
            print("Apply testing transform.")
            return joint_augmentation.Compose([
                joint_augmentation.ToTensor(),
            ])
        
    def normalize(self, data):
        data = self.sample_clip(data)
        data = self.normalize_all(data)
        return data
    
    def normalize_all(self, data):
        for k, v in data.items():
            if k == 'rgb' or k == 'depth':
                data[k] = self.video_norm(v.float())
            elif k == '2d_skeleton':
                data[k] = self.shoulder_hand_norm(v)
            elif k == 'r_features' or k == 'd_features':
                continue
            else:
                raise ValueError(f'unknown data type {k}')
        return data
    
    def video_norm(self, video):

        # for video mae norm
        video /= 255

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        for i in range(3):
            video[:,i,...] -= mean[i]
            video[:,i,...] /= std[i]

        return video

    def fix_norm(self, origin_input_data):
        confidence_flag = True if origin_input_data.shape[-1] == 3 else False
        if confidence_flag:
            conf = origin_input_data[ :, :, -1]
            input_data = origin_input_data[ :, :, :-1]
        else:
            input_data = origin_input_data
        input_data = input_data / 256
        input_data = self.centeralize(input_data)

        if confidence_flag:
            return torch.cat([input_data, conf.unsqueeze(-1)], dim=-1)
        else:
            return input_data

    def shoulder_hand_norm(self, origin_input_data):
        confidence_flag = True if origin_input_data.shape[-1] == 3 else False
        if confidence_flag:
            conf = origin_input_data[ :, :, -1]
            input_data = origin_input_data[ :, :, :-1]
        else:
            input_data = origin_input_data

        if 'body' in self.module_keys:
            shoulder_length = torch.mean(torch.sqrt(torch.sum(torch.pow(input_data[:,3,:] - input_data[:,4,:], 2), dim=-1)))
            # norm according to shoulder, as person will be close or far to the screen
            input_data = input_data / (shoulder_length * 3)

        # hand norm
        # for left hand
        if 'left_hand' in self.module_keys:
            left_hand_range = self.kps_info['left_hand']['kps_rel_range']
            if 'body' in self.module_keys:
                input_data[:,left_hand_range[0]:left_hand_range[1]] *= (shoulder_length * 3) # recover
            left_hand_length = torch.sqrt(((input_data[:,(left_hand_range[0]+9)] - input_data[:,left_hand_range[0]]) ** 2).sum(-1)).mean()
            input_data[:,left_hand_range[0]:left_hand_range[1]] /= (left_hand_length * 2)

        # for right hand
        if 'right_hand' in self.module_keys:
            right_hand_range = self.kps_info['right_hand']['kps_rel_range']
            if 'body' in self.module_keys:
                input_data[:,right_hand_range[0]:right_hand_range[1]] *= (shoulder_length * 3) # recover
            right_hand_length = torch.sqrt(((input_data[:,(right_hand_range[0]+9)] - input_data[:,right_hand_range[0]]) ** 2).sum(-1)).mean()
            input_data[:,right_hand_range[0]:right_hand_range[1]] /= (right_hand_length * 2)
        
        input_data = input_data - 1
        input_data = self.centeralize(input_data)

        if confidence_flag:
            return torch.cat([input_data, conf.unsqueeze(-1)], dim=-1)
        else:
            return input_data
    
    def centeralize(self, input_data):
        for k, v in self.kps_info.items():
            kps_rng = v['kps_rel_range']
            if k == 'body':
                input_data[:, kps_rng[0]: kps_rng[1]] = (
                    input_data[:, kps_rng[0]: kps_rng[1]] - input_data[0, v['norm_kps_rel_index']].mean(0)[None, None]
                )
            else:
                input_data[:, kps_rng[0]: kps_rng[1]] = (
                    input_data[:, kps_rng[0]: kps_rng[1]] - input_data[:, v['norm_kps_rel_index']]
                )
        return input_data

    def __len__(self):
        return len(self.inputs_list)
    
    @staticmethod
    def collate_fn(batch,temporaltype): 
        def padded(in_data,res_frames): # in_data.shape: [T,51,3]
            ret = torch.cat( # since the sliding starts from the left edge so the left side is not padded
                (
                    in_data,
                    in_data[-1][None].expand(res_frames, -1, -1),
                ),
                dim=0,
            )
            return ret
        
        def padded_ft(in_data,res_frames): # in_data.shape: [window_num,768]
            ret = torch.cat( # since the sliding starts from the left edge so the left side is not padded
                (
                    in_data,
                    in_data[-1][None].expand(res_frames, -1),
                ),
                dim=0,
            )
            return ret
        def getmiddle(ft): # for single-modal testing, 32 / 16 = 2 windows
            len = ft.shape[0]
            if len == 2:
                return ft
            if len == 1:
                return ft.repeat(2,1)
            mid = len // 2
            return ft[mid-1:mid+1]
        
        batched_data = {'data': {}}
        for k in batch[0].keys():
            if k == 'label':
                batched_data[k] = torch.LongTensor([item[k] for item in batch])
            elif k == 'info':
                batched_data[k] = [item[k] for item in batch]
            else:
                if temporaltype == 'LSTM': # only padded for convenient fetch
                    mx_len = max([len(item[k]) for item in batch]) 
                    if k == '2d_skeleton':
                        batched_data['len'] = torch.tensor([len(item[k]) for item in batch])
                        batched_data['data'][k] = torch.stack(
                        [padded(item[k],mx_len-len(item[k])) for item in batch]
                        , dim=0)
                    else:
                        if k == 'r_features':
                            batched_data['rgb_len'] = torch.tensor([len(item[k]) for item in batch])
                        else: # d_features
                            batched_data['depth_len'] = torch.tensor([len(item[k]) for item in batch])
                        batched_data['data'][k] = torch.stack(
                            [padded_ft(item[k],mx_len-len(item[k])) for item in batch]
                            , dim=0)
                else: # single modality
                    if k == '2d_skeleton':
                        batched_data['data'][k] = torch.stack([item[k] for item in batch], dim=0)
                    else: # features
                        batched_data['data'][k] = torch.stack(
                            [getmiddle(item[k]) for item in batch]
                            , dim=0)
        return batched_data
