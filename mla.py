import torch
from torch import Tensor
from torch import nn

import math

from .config import DeepSeekConfig

def calculate_cos_sin(seq_len, dim, device, base=10_000.0):
    position = torch.arange(0, seq_len, device=device, dtype=torch.float).unsqueeze(0)
    ids = torch.arange(0, dim // 2, device=device, dtype=torch.float)
    
    theta =  base ** -(2 * ids) / (dim // 2)
    rope = torch.matmul(position.transpose(0, 1), theta.unsqueeze(0)).to(device)
    
    return rope.cos(), rope.sin()
    
def rope(x:Tensor, cos:Tensor, sin:Tensor): 
    b, seq, d_model = x.shape 
    x = x.view(b, seq, d_model//2, 2) 
    out = torch.stack([ 
        x[..., 0] * cos - x[..., 1] * sin, 
        x[..., 1] * cos + x[..., 0] * sin 
    ],dim=-1) 
    out = out.contiguous().view(b, seq, d_model) 
    return out

class Attention(nn.Module):
    def __init__(self, config:DeepSeekConfig):
        super().__init__()
        
        self.head_dim = config.d_model // config.num_head
        self.num_head = config.num_head
        
        self.Wd_q = nn.Linear(config.d_model, config.d_q)
        self.Wu_q = nn.Linear(config.d_q, config.d_model)
        
        self.Wq_r = nn.Linear(config.d_q, config.d_model)
        self.Wk_r = nn.Linear(config.d_model, config.d_model)
        
        self.Wd_kv = nn.Linear(config.d_model, config.d_kv)
        self.Wu_k = nn.Linear(config.d_kv, config.d_model)
        self.Wu_v = nn.Linear(config.d_kv, config.d_model)
        
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        self.scale = math.sqrt(config.num_head + config.d_kv + config.d_q)
        
    def forward(self, x:Tensor) -> Tensor:
        
        b, seq_len, d_model = x.shape
        
        ct_q = self.Wd_q(x) #(b, seq, d_q)
        qt_c = self.Wu_q(ct_q) #(b, seq, d_model)
        qt_c = qt_c.view(b, seq_len, self.num_head, self.head_dim).transpose(-3, -2) #(b, num_head, seq, head_dim)
        
        cos_sin = calculate_cos_sin(seq_len, d_model, x.device)
        qt_r = rope(self.Wq_r(ct_q), *cos_sin).view(b, seq_len, self.num_head, self.head_dim).transpose(-3, -2) #(b, num_head, seq, head_dim)
        
        query = qt_c + qt_r # (b, num_head, seq, head_dim)
        
        ct_kv = self.Wd_kv(x)
        kt_c = self.Wu_k(ct_kv).view(b, seq_len, self.num_head, self.head_dim).transpose(-3, -2) #(b,num_head, seq, head_dim)
        
        cos_sin = calculate_cos_sin(seq_len, d_model, x.device)
        kt_r = rope(self.Wk_r(x), *cos_sin).view(b, seq_len, self.num_head, self.head_dim).transpose(-3, -2) #(b, num_head, seq, head_dim)
        
        key = kt_c + kt_r #(b, num_head, seq, head_dim)
        
        value = self.Wu_v(ct_kv).view(b, seq_len, self.num_head, self.head_dim).transpose(-3, -2) #(b, num_head, seq, head_dim)
        
        attention = torch.matmul(query, key.transpose(-2, -1)) / self.scale #(b, num_head, seq, seq)
        attention = attention.softmax(dim=-1) #(b, num_head, seq, seq)
        
        out = torch.matmul(attention, value) #(b, num_head, seq, head_dim)
        out = out.transpose(-3, -2).contiguous().view(b, seq_len, self.num_head * self.head_dim) #(b, seq_len, d_model)
        out = self.out_proj(out) #(b, seq_len, d_model)
        
        return out