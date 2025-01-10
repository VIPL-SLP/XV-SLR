import pdb
import torch
import modules
import numpy as np
import torch.nn as nn
from utils import Decode
from torch.nn import init
from torch.nn import Parameter
import torch.nn.functional as F

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class NormBothLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormBothLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        outputs = torch.matmul(F.normalize(x, dim=-1), F.normalize(self.weight, dim=0))
        return outputs


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        y = self.classifier(x)

        return y
        

class NormalizedClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5) 

    def forward(self, x):
        w = self.weight  

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)
    

def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training
    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False