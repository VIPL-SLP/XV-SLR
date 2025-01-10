import os
import pdb
import torch
import modules
import torch.nn as nn
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Conv2d(nn.Module):
    def __init__(self, module_type, **module_kwargs):
        super().__init__()
        if 'torch_home' in module_kwargs.keys():
            os.environ['TORCH_HOME'] = module_kwargs.pop('torch_home')
        self.conv2d = getattr(models, module_type)(**module_kwargs)
        self.module_type = module_type
        if module_type == 'resnet18':
            self.conv2d.fc = Identity()

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, feats, feat_lgt):
        if self.module_type == 'resnet18':
            batch, temp, channel, height, width = feats.shape
            inputs = feats.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, feat_lgt)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            return {
                'feats': framewise,
                'len_x': feat_lgt,
            }
        return self.conv2d(feats, feat_lgt)