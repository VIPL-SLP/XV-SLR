
import sys
import pdb
import torch

import numpy as np
from PIL import Image
import torch.utils.data as data
from utils import video_augmentation
from .dataloader_base import BaseFeeder

sys.path.append("..")

class VideoFeeder(BaseFeeder):
    def __init__(
            self, gloss_dict, inputs_list, data_type='video', 
            mode="train", transform_mode=True, isolated=False, dry_run=False,
            ):
        super().__init__(
            gloss_dict, inputs_list, data_type, 
            mode, transform_mode, isolated
            )
        if dry_run:
            self.inputs_list = self.inputs_list[:100]
        self.spatial_data_aug = self.spatial_transform()
        self.temporal_data_aug = self.temporal_transform()

    def read_data(self, info):
        # load video
        st_idx = info['start_idx']
        video = [self._load_image(info['img_regular'].format(i+st_idx)) for i in range(info['num_frames'])]
        return video

    def normalize(self, video):
        print("Should not reach here!")
        video = video.float() / 127.5 - 1
        return video

    def spatial_transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            print("Should not reach here!")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.ToTensor(),
            ])

    def temporal_transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                video_augmentation.TemporalRescale(0.2),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.ToTensor(),
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
        return len(self.inputs_list) - 1