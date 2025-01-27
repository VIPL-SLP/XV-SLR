import torch
import torch.nn as nn
import torch.nn.functional as F

class NormBothLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormBothLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(F.normalize(x, dim=-1), F.normalize(self.weight, dim=0))
        return outputs