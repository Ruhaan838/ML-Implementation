import torch
from torch import nn
from torch.nn import functional as F


class DyT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.alpha = nn.Parameter(torch.ones(1) * config.init_alpha)
        self.beta = nn.Parameter(torch.zeros(config.d_model))
        self.gamma = nn.Parameter(torch.ones(config.d_model))
        
    def forward(self, x: torch.Tensor):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta
        
    