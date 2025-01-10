import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import reduce

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def length2mask(feat_len, device):
    max_length = torch.max(feat_len)
    attention_mask = torch.zeros([feat_len.shape[0], max_length], dtype=torch.long, device=device)
    for i in range(len(feat_len)):
        attention_mask[i, :feat_len[i]] = 1
    return attention_mask

class Filip_loss(nn.Module):
    def __init__(self, adaptive_temp=False, temp=0.07) -> None:
        super().__init__()
        if adaptive_temp:
            self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temp))
        else:
            self.temperature = torch.ones([]) * np.log(1 / temp)
    
    def forward(self, vision, text, vision_mask, text_mask):
        temp = self.temperature.exp()
        sim_text_to_image = einsum('x t d, y i d -> x y t i', text, vision) * temp
        # cal for text
        mask_sim = sim_text_to_image.masked_fill(~vision_mask[None,:,None,:], max_neg_value(sim_text_to_image.dtype))
        text_to_image = reduce(mask_sim, '... t i -> ... t', 'max')
        text_to_image = masked_mean(text_to_image, text_mask[:,None,:], dim=-1)
        # cal for vision
        mask_sim = sim_text_to_image.masked_fill(~text_mask[:,None,:,None], max_neg_value(sim_text_to_image.dtype))
        image_to_text = reduce(mask_sim, '... t i -> ... i', 'max')
        image_to_text = masked_mean(image_to_text, vision_mask[None,:,:], dim=-1)

        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))
        text_to_image_pos, image_to_text_pos = map(torch.diag, (text_to_image_exp, image_to_text_exp))
        text_to_image_denom = text_to_image_exp.sum(-1)
        image_to_text_denom = image_to_text_exp.sum(0)

        text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim = -1)
        image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim = -1)

        return {
            'filip_loss': (text_to_image_loss + image_to_text_loss) / 2,
            'text2image': text_to_image_loss,
            'image2text': image_to_text_loss
        }