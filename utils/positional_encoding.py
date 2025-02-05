import torch
import math
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # -> [B, T, C]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, T, C]
        x = x + self.pe[:, :x.size(1)]
        return x
