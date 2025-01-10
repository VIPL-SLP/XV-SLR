import os
import pdb
import torch
import modules
import torch.nn as nn
import torchvision.models as models
from modules import gcn_modules

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def convert_model(module, momentum=0.2):
    mod = module
    for pth_module, sync_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                        torch.nn.modules.batchnorm.BatchNorm2d,
                                        torch.nn.modules.batchnorm.BatchNorm3d],
                                       [torch.nn.modules.batchnorm.BatchNorm1d,
                                        torch.nn.modules.batchnorm.BatchNorm2d,
                                        torch.nn.modules.batchnorm.BatchNorm3d]):
        if isinstance(module, pth_module):
            mod = sync_module(module.num_features, module.eps, momentum, module.affine)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child, momentum))
    return mod

class STGCN(nn.Module):
    def __init__(self, module_type, **module_kwargs):
        super().__init__()
        self.module_type = module_type
        # bn_momentum = module_kwargs.pop('bn_momentum', 0.1)
        self.gcn = getattr(gcn_modules, module_type)(**module_kwargs)
        # convert_model(self.gcn, momentum=bn_momentum)

    def forward(self, feats):
        if self.module_type == 'CoSign1s':
            # batch, temp, kps, channel = feats.shape
            framewise = self.gcn(feats)
            feat_key = 'fusion' if 'fusion' in framewise.keys() else 'view1'
            return framewise[feat_key] # B,T,C
        else:
            raise ValueError('unknown module type')
