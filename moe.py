import torch
from torch import nn
from torch.nn import functional as F

from config import GPTConfig


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        return F.softmax(self.linear2(x), dim=1)
    
class Gating(nn.Module):
    def __init__(self, d_model, num_expert):
        super().__init__()
        
        self.linear = nn.Linear(d_model, num_expert)
        
    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)
    
class MoE(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        
        self.top_k = config.top_k
        self.experts = nn.ModuleList([Expert(config.d_model, config.d_ff) for _ in range(config.num_expert)])
        self.gate = Gating(config.d_model, config.num_expert)
    
    def forward(self, x):
        B, S, D = x.shape
        x_f = x.view(B*S, D)
        weigths = self.gate(x)
        topk_val, topk_ind = torch.topk(weigths, self.top_k)
        
        topk_val_f, topk_ind_f = topk_val.view(B*S, -1), topk_ind.view(B*S, -1)
        
        out = torch.zeros(B*S, self.top_k, D).to(x.device)
        
        for ind in range(self.top_k):
            mask = (topk_ind_f == ind)
            if not mask.any():
                continue
            
            t_ind, slot = torch.where(mask)
            
            exp_in = x_f[t_ind]
            exp_out = self.experts[ind](exp_in)
            
            out[t_ind, slot] = exp_out

        out *= topk_val_f.unsqueeze(-1)
        out = out.sum(dim=1)
        out = out.view(B, S, D)
        
        return out
        
    
if __name__ == "__main__":
    config = GPTConfig(d_model=12, head_size=6, num_expert=5, top_k=2)
    
    a = torch.rand(2, 3, 12)
    m = MoE(config)
    print(m(a).shape)
    