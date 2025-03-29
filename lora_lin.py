import torch
from torch import nn
from torch.nn import functional as F

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, lora_rank, alpha=1.0, dropout=0.0):
        super().__init__()
        
        self.alpha = alpha
        self.lora_rank = lora_rank
        
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.weights.requires_grad = False
        
        self.A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, lora_rank) * 0.01)
        
        self.scaling = alpha / lora_rank
        
        self.dropout = nn.Dropout(dropout)
        
             
    def forward(self, x):
        lora_w = F.linear(self.dropout(x), self.A)
        lora_w = F.linear(lora_w, self.B)
        out = F.linear(x, self.weights, self.scaling * lora_w)
        return out
    
if __name__ == "__main__":
    lora = LoraLinear(10, 5, 2, dropout=0.01)
    x = torch.randn(1, 10)
    print(lora(x).shape)