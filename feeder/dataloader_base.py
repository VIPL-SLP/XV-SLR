
import sys
import pdb
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation

sys.path.append("..")

class BaseFeeder(data.Dataset):
    def __init__(
            self, gloss_dict, inputs_list, data_type='video', 
            mode="train", transform_mode=True, isolated=False,
            ):
        self.mode = mode
        self.dict = gloss_dict
        self.isolated = isolated
        self.data_type = data_type
        self.inputs_list = inputs_list
        self.transform_mode = "train" if transform_mode else "test"
        print(mode, len(self))
        self.spatial_data_aug = self.spatial_transform()
        self.temporal_data_aug = self.temporal_transform()
        print("")

    def __getitem__(self, idx):
        ret_dict = dict()
        data, label, fi = self.read_item(idx)
        ret_dict[self.data_type] = data
        ret_dict['label'] = torch.LongTensor(label)
        ret_dict['gloss'] = self.inputs_list[idx]['gloss_sequence']
        ret_dict['text'] = self.inputs_list[idx]['text']
        ret_dict['info'] = self.inputs_list[idx]
        ret_dict['isolated'] = self.isolated
        return ret_dict

    def read_item(self, idx, state=None):
        # load file info
        fi = self.inputs_list[idx]
        label_list = []
        if self.isolated:
            split_gloss = [fi['gloss_sequence']]
        else:
            split_gloss = fi['gloss_sequence'].split(" ")
        for gloss in split_gloss:
            if gloss == '':
                continue
            if gloss in self.dict['gloss2id'].keys():
                label_list.append(self.dict['gloss2id'][gloss]['index'])
        data = self.read_data(fi)
        data = self.spatial_data_aug(data)
        if state is not None:
            random.setstate(state)
        data = self.temporal_data_aug(data)
        data = self.normalize(data)
        return data, label_list, fi

    def _load_image(self, img_path):
        return np.array(Image.open(img_path).convert('RGB'))

    def read_data(self, info):
        raise NotImplementedError("")

    def spatial_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def temporal_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    def normalize(self, data):
        raise NotImplementedError("")

    def __len__(self):
        return len(self.inputs_list) - 1
