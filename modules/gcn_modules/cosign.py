import pdb
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .gcn_utils import Graph
from .st_gcn_block import get_stgcn_chain

def generate_mask(shape, part_num, clip_length, ratio, dim): # clip_length = 25
    B, T, C = shape
    clips = T // clip_length
    random_mask = np.random.rand(B, clips, part_num) > (1 - 2 * ratio)
    mask_q, mask_k = np.zeros_like(random_mask), np.zeros_like(random_mask)
    position = np.where(random_mask)
    half_num = int(len(position[0]) / 2)

    index = np.random.choice(len(position[0]), half_num, replace=False).tolist()
    for i in range(len(position[0])):
        if i in index:
            mask_q[position[0][i], position[1][i], position[2][i]] = 1
        else:
            mask_k[position[0][i], position[1][i], position[2][i]] = 1
    mask_q = mask_q.astype(np.bool8)
    mask_k = mask_k.astype(np.bool8)

    mask_cat_q = torch.ones(shape)
    mask_cat_k = torch.ones(shape)
    for i in range(B):
        for k in range(clips):
            if k == clips - 1:
                for j in range(mask_q.shape[2]):
                    if mask_q[i, k, j]:
                        mask_cat_q[i, clip_length*k:, dim * j : dim * (j + 1)] = 0
                    if mask_k[i, k, j]:
                        mask_cat_k[i, clip_length*k:, dim * j : dim * (j + 1)] = 0
            else:
                for j in range(mask_q.shape[2]):
                    if mask_q[i, k, j]:
                        mask_cat_q[i, clip_length*k:clip_length*(k+1), dim * j : dim * (j + 1)] = 0
                    if mask_k[i, k, j]:
                        mask_cat_k[i, clip_length*k:clip_length*(k+1), dim * j : dim * (j + 1)] = 0
    return mask_cat_q, mask_cat_k


class CoSign1s(nn.Module):
    def __init__(self, 
                 in_channels, temporal_kernel, hidden_size, 
                 level, kps_config=None, adaptive=True, CR_args=None,
                 cat_hand=True
                 ) -> None:
        super().__init__()
        self.graph, A = {}, {}
        self.gcn_modules = {}
        self.in_channels = in_channels
        self.CR_args = CR_args
        self.cat_hand = cat_hand
        with open(kps_config, 'r') as f:
            self.kps_info = yaml.load(f, Loader=yaml.FullLoader)
        self.part_num = len(self.kps_info)
        
        self.linear = nn.Sequential(
            nn.Linear(in_channels, 64),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        for _, v in self.kps_info.items():
            graph_mode = v['graph_mode']
            module_key = v['module_key']
            self.graph[module_key] = Graph(layout=f'custom_{graph_mode}', strategy='distance', max_hop=1)
            A[module_key] = torch.tensor(self.graph[module_key].A, dtype=torch.float32, requires_grad=False)
            spatial_kernel_size = A[module_key].size(0)
            self.gcn_modules[module_key], final_dim = get_stgcn_chain(
                64, level, (temporal_kernel, spatial_kernel_size), 
                A[module_key].clone(), adaptive
            )

        self.pool_func = F.avg_pool2d
        self.gcn_modules = nn.ModuleDict(self.gcn_modules)
        self.fusion = nn.Sequential(nn.Linear(final_dim*self.part_num, hidden_size), nn.ReLU(inplace=True))
        self.out_size = hidden_size
        self.final_dim = final_dim
    
    def process_part_features(self, features, cat_hand_flag=False):
        feat_list = []
        for k, v in self.kps_info.items():
            module_key = v['module_key']
            kps_rng = v['kps_rel_range']
            if 'left_hand' in self.kps_info.keys() and 'right_hand' in self.kps_info.keys():
                # print("success!!")
                if cat_hand_flag:
                    if k == 'left_hand':
                        kps_rng_right = self.kps_info['right_hand']['kps_rel_range']
                        cat_hand = torch.cat(
                            [
                                features[..., kps_rng[0]:kps_rng[1]],
                                features[..., kps_rng_right[0]:kps_rng_right[1]]
                            ], dim=0
                        )
                        part_feat = self.gcn_modules[module_key](cat_hand)
                        pooled_feat = self.pool_func(part_feat, (1, kps_rng[1] - kps_rng[0])).squeeze(-1)
                        feat_list += list(torch.chunk(pooled_feat, 2, dim=0))
                        continue
                    if k == 'right_hand':
                        continue
            part_feat = self.gcn_modules[module_key](features[..., kps_rng[0]: kps_rng[1]])
            pooled_feat = self.pool_func(part_feat, (1, kps_rng[1] - kps_rng[0])).squeeze(-1)
            feat_list.append(pooled_feat)
        return torch.cat(feat_list, dim=1)
    
    def forward(self, x):
        if x.shape[3] == 7:
            static = torch.cat([x[..., 0:2], x[..., 6].unsqueeze(-1)], dim=-1)
        else:
            static = x
        static = static[..., :self.in_channels]
        # linear stage static.shape: [B(N), T, 55(V), 3(C)]
        static = self.linear(static).permute(0, 3, 1, 2) # N, C, T, V permute: 换位
        cat_feat = self.process_part_features(static, self.cat_hand).transpose(1, 2) # [B,T,C]

        if self.CR_args is not None and self.training:
            mask_view1, mask_view2 = generate_mask(cat_feat.shape, self.part_num, \
                                    self.CR_args['clip_length'], self.CR_args['ratio'], self.final_dim)
            view1, view2 = mask_view1.to(cat_feat.device) * cat_feat, mask_view2.to(cat_feat.device) * cat_feat
            # view1, view2 = self.fusion(view1), self.fusion(view2)
            view1 = self.fusion(view1)
            return {
                'view1': view1,
                # 'view2': view2
            }
        else:
            fusion_feat = self.fusion(cat_feat)
            return {
                'cat': cat_feat,
                'fusion': fusion_feat
            }