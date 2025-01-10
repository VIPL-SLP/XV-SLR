
import sys
import pdb
import yaml
import torch
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data
from utils import skeleton_augmentation
from .dataloader_base import BaseFeeder

sys.path.append("..")

class PoseFeeder(BaseFeeder):
    def __init__(
            self, 
            gloss_dict, 
            inputs_list, 
            kps_config=None,
            data_type='pose', 
            mode="train", 
            dry_run=False,
            transform_mode=True, 
            isolated=False,
            num_frames=-1, 
            num_classes=-1,
            ):
        super().__init__(
            gloss_dict, inputs_list, data_type, 
            mode, transform_mode, isolated
            )
        
        if num_classes != -1:
            self.inputs_list = [*filter(
                lambda x: gloss_dict['gloss2id'][x['gloss_sequence']]['index'] < num_classes, 
                self.inputs_list
                )]
            print(f"filtered: {len(self.inputs_list)}")

        if dry_run:
            self.inputs_list = self.inputs_list[:100]
        with open(kps_config, 'r') as f:
            self.kps_info = yaml.load(f, Loader=yaml.FullLoader)

        self.num_frames = num_frames
        self.pose_idx = sum([v['kps_index'] for k, v in self.kps_info.items()], [])
        self.spatial_data_aug = self.spatial_transform()
        self.temporal_data_aug = self.temporal_transform()

    def read_data(self, info):
        info['skeleton_path'] = info['skeleton_path'].replace('slt_phoenix2014T_hrnet/fullFrame-256x256px', 'phoenix14T_merge_skeleton')
        info['skeleton_path'] = info['skeleton_path'][:-4] + '.npy'
        # load pose
        if info['skeleton_path'].endswith('pkl'):
            pose = pickle.load(open(info['skeleton_path'], "rb"))
        elif info['skeleton_path'].endswith('npy'):
            pose = np.load(info['skeleton_path'])
        else:
            assert("Unknown error.")
        return pose[:, self.pose_idx]
    
    def sample_clip(self, pose_seq):
        lgt = len(pose_seq)
        if self.transform_mode == "train":
            if lgt >= self.num_frames:
                start_frame = np.random.randint(0, max(lgt-self.num_frames, 1))
                clip_idx = [i for i in range(start_frame, start_frame + self.num_frames)]
            else:
                res_frames = self.num_frames - lgt
                lpad = np.random.randint(0, max(res_frames, 1))
                rpad = res_frames - lpad
                clip_idx = [0] * lpad + [i for i in range(lgt)] + [lgt-1] * rpad
        else:
            if lgt >= self.num_frames:
                start_frame = (lgt-self.num_frames) // 2
                clip_idx = [i for i in range(start_frame, start_frame + self.num_frames)]
            else:
                res_frames = self.num_frames - lgt
                lpad = res_frames // 2
                rpad = res_frames - lpad
                clip_idx = [0] * lpad + [i for i in range(lgt)] + [lgt-1] * rpad
        return pose_seq[clip_idx]

    def normalize(self, input_data):
        print("Should not reach here!")
        if self.num_frames != -1:
            input_data = self.sample_clip(input_data)
        pose_data = self.uniform_split_pose_normalize_hand21(input_data)
        return pose_data
    
    def uniform_split_pose_normalize_hand21(self, origin_input_data):
        confidence_flag = True if origin_input_data.shape[-1] == 3 else False
        if confidence_flag:
            conf = origin_input_data[ :, :, -1]
            input_data = origin_input_data[ :, :, :-1]
        else:
            input_data = origin_input_data

        # width = 127.5
        width = abs(input_data[:, 3] - input_data[:, 4])[:, 0].mean().item() * 2
        # input_data = input_data / max(1, width)
        if (input_data / max(1, width)).max() > 5:
            input_data = input_data / 160
        else:
            input_data = input_data / max(1, width)
        # print(input_data.min(), input_data.max())

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

        if confidence_flag:
            return torch.cat([input_data, conf.unsqueeze(-1)], dim=-1)
        else:
            return input_data

    def spatial_transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return skeleton_augmentation.Compose([
                # skeleton_augmentation.RandomHorizontalFlip(0.5),
                # skeleton_augmentation.RandomResize(0.2),
                # skeleton_augmentation.RandomRot(0.1 * np.pi),
                skeleton_augmentation.ToTensor(),
            ])
        else:
            print("Apply testing transform.")
            return skeleton_augmentation.Compose([
                skeleton_augmentation.ToTensor(),
            ])

    def temporal_transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return skeleton_augmentation.Compose([
                skeleton_augmentation.TemporalRescale(0.2),
            ])
        else:
            print("Apply testing transform.")
            return skeleton_augmentation.Compose([
                skeleton_augmentation.ToTensor(),
            ])

    @staticmethod
    def collate_fn(batch):
        if batch[0]['isolated']:
            left_pad = 0
            total_pad = 0
        else:
            left_pad = 6
            total_pad = 12
        sorted_key = (set(batch[0].keys()) & set(['pose', 'video', 'feat'])).pop()
        batch = [item for item in sorted(batch, key=lambda x: len(x[sorted_key]), reverse=True)]
        batch_data = dict()

        for key in batch[0].keys():
            if key in ['gloss', 'text', 'info']:
                batch_data[key] = [item[key] for item in batch]
            elif key == 'isolated':
                batch_data.pop(key, None)
            elif key == 'label':
                label = [item[key] for item in batch]
                padded_label = []
                label_length = torch.LongTensor([len(lab) for lab in label])
                for lab in label:
                    padded_label.extend(lab)
                padded_label = torch.LongTensor(padded_label)
                batch_data['label'] = padded_label
                batch_data['label_lgt'] = label_length
            else:
                seq = [item[key] for item in batch]
                length = torch.LongTensor(
                    [np.ceil(len(s) / 4.0) * 4 + total_pad for s in seq]
                )
                max_len = max(length)
                expand_dim = seq[0][0][None].shape[1:]
                padded_seq = [
                    torch.cat(
                        (
                            s[0][None].expand((left_pad,) + expand_dim),
                            s,
                            s[-1][None].expand((max_len-len(s)-left_pad,) + expand_dim),
                        ),
                        dim=0,
                    )
                    for s in seq
                ]
                padded_seq = torch.stack(padded_seq)
                batch_data[key] = padded_seq
                batch_data[f'{key}_lgt'] = length
        return batch_data

    def __len__(self):
        return len(self.inputs_list)
