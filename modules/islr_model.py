import pdb
import torch
import os
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from modules import visual_extractor as VEncoder
from modules import heads as Heads
from modules.temporal_module import BiLSTMLayer
from modules.temporal_module.transformer import TransformerEncoderLayer, TransformerEncoder
from utils.positional_encoding import PositionalEncoding

class ISLRModel(nn.Module):
    def __init__(self, visual_backbone_args, head_args, temporal_arg, loss_weight, pred_head, class_num, weights=None) -> None:
        super().__init__()
        self.temporal_arg = temporal_arg
        self.temporaltype = temporal_arg['type']
        visual_backbone = {}
        for data_type, v_args in visual_backbone_args.items():
            backbone_type = v_args.pop('type')
            visual_backbone[data_type] = getattr(VEncoder, backbone_type)(**v_args)
        self.visual_backbone = nn.ModuleDict(visual_backbone)

        if self.temporaltype == 'LSTM':
            self.contextual_module = BiLSTMLayer(rnn_type='LSTM', input_size=1024, hidden_size=1024,
                                            num_layers=2, bidirectional=True)
        heads = {}
        for name, h_args in head_args.items():
            head_type = h_args.pop('type')
            if 'class_num' in h_args.keys():
                h_args['class_num'] = class_num
            heads[name] = getattr(Heads, head_type)(**h_args)
        self.heads = nn.ModuleDict(heads)

        self.loss_weight = loss_weight
        if weights is not None:
            self.load_weights(weights)
        self.pred_head = pred_head
    
    def load_weights(self, weights):
        # load weights for single modality here
        pass

    def to_device(self, data, device):
        for k in data.keys():
            data[k] = data[k].to(device.output_device)
        return data
    
    def pad_rgb_feat_list(self,rgb_feat_list):
        def padded_ft(in_data,res_frames): # in_data.shape: [window_num,768]
            ret = torch.cat( # 左边不填充，因为要从左往右滑动
                (
                    in_data,
                    in_data[-1][None].expand(res_frames, -1),
                ),
                dim=0,
            )
            return ret
        max_t = max(len(r) for r in rgb_feat_list)
        return torch.stack(
                        [padded_ft(torch.from_numpy(r),max_t-len(r)) for r in rgb_feat_list]
                        , dim=0)

    def forward(self, batch_data, device, **kwargs):
        feat_dict = {}
        data = batch_data['data']
        data = self.to_device(data, device)
        label = batch_data['label'].to(device.output_device)
        len_x = None
        if self.temporaltype == 'LSTM':
            len_x = batch_data['len']
            for data_type in data.keys(): # ['2d_skeleton']
                if data_type == 'r_features':
                    feat_dict['rgb'] = data['r_features'].to(device.output_device)
                    continue
                if data_type == 'd_features':
                    feat_dict['depth'] = data['d_features'].to(device.output_device)
                    continue
                feat_dict[data_type] = self.visual_backbone[data_type](data[data_type]) # [B, mx_len, 1024] 
                
                # input: [B, mx_len, V, feat_out_dim(1024)]
                # output: [B, mx_len, feat_out_dim]
                contextual_ret = self.contextual_module(feat_dict[data_type].transpose(0,1), len_x)['predictions'].transpose(0,1)
                
                feat_dict[data_type] = contextual_ret # for per-frame loss
        else: # fixed num_frames(64)
            for data_type in data.keys(): # ['2d_skeleton']
                if data_type == 'r_features':
                    feat_dict['rgb'] = data['r_features'].to(device.output_device)
                    continue
                if data_type == 'd_features':
                    feat_dict['depth'] = data['d_features'].to(device.output_device)
                    continue
                    
                feat_dict[data_type] = self.visual_backbone[data_type](data[data_type]) # [B, 64, 1024]
        # pdb.set_trace()
        # feat_dict[data_type] -> [B, T, C] or [B,C] if not using loc_loss
        total_loss, total_loss_details = 0, {}
        if 'r_features' in data.keys() and 'd_features' in data.keys(): # RGB-D
            if self.training:
                for k in self.heads.keys():
                    ret = self.heads[k](feat_dict, len_x,batch_data['rgb_len'],batch_data['depth_len'],device.output_device,label,len_x,temporal_arg=self.temporal_arg)
                    total_loss += self.loss_weight[k] * ret['loss']
                    total_loss_details.update(ret['loss_details'])
                return {
                    'loss': total_loss,
                    'loss_details': total_loss_details
                }
            else:
                return self.heads[self.pred_head](feat_dict,len_x,batch_data['rgb_len'],batch_data['depth_len'], device.output_device,label,len_x,temporal_arg=self.temporal_arg)
        elif 'r_features' in data.keys() and '2d_skeleton' in data.keys(): # RGB
            if self.training:
                for k in self.heads.keys():
                    ret = self.heads[k](feat_dict, len_x,batch_data['rgb_len'],None,device.output_device,label,len_x,temporal_arg=self.temporal_arg)
                    total_loss += self.loss_weight[k] * ret['loss']
                    total_loss_details.update(ret['loss_details'])
                return {
                    'loss': total_loss,
                    'loss_details': total_loss_details
                }
            else:
                return self.heads[self.pred_head](feat_dict,len_x,batch_data['rgb_len'],None, device.output_device,label,len_x,temporal_arg=self.temporal_arg)
        else: # single modal
            if self.training:
                for k in self.heads.keys():
                    ret = self.heads[k](feat_dict, label,len_x,temporal_arg=self.temporal_arg)
                    total_loss += self.loss_weight[k] * ret['loss']
                    total_loss_details.update(ret['loss_details'])
                return {
                    'loss': total_loss,
                    'loss_details': total_loss_details
                }
            else:
                return self.heads[self.pred_head](feat_dict, label,len_x,temporal_arg=self.temporal_arg)
