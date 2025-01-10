import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=('K5', 'P2', 'K5', 'P2'), use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=int(ks[1])//2)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
            elif ks[0] == 'C':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def update_lgt(self, lgt):
        # feat_len = copy.deepcopy(lgt)
        feat_len = lgt
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, int(ks[1])).long()
            # Add padding and remove the length reduction
            elif ks[0] == 'C':
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
