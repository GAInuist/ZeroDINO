# Module - LCAR

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
Norm = nn.LayerNorm

class CrossAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8,
                 attn_drop=0.1, proj_drop=0.1, bias=True):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=bias,
            batch_first=True
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        """
        q: [B, Q_len, D]
        k: [B, K_len, D]
        v: [B, K_len, D]
        """
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        out, attn_weights = self.attn(q, k, v, need_weights=True,
                                      average_attn_weights=False)
        out = self.proj_drop(self.proj(out))
        return out, attn_weights      # out [B, Q_len, D]; attn_weights [B, heads, Q_len, K_len]

class MLP(nn.Module):
    def __init__(self, dim, drop_rate=0.1, mlp_ratio=2):
        super(MLP, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Local_Refiner(nn.Module):
    def __init__(self, attr_num, dim=768):
        super(Local_Refiner, self).__init__()
        self.dim = dim
        self.attr_num = attr_num
        self.attn = CrossAttention(dim=self.dim, num_heads=8, attn_drop=0.1, proj_drop=0.1, bias=True)
        self.mlp = MLP(dim=self.dim, drop_rate=0.1, mlp_ratio=2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, local_tokens, w2v):
        refined_local, attn_mask = self.attn(local_tokens, w2v, w2v)
        local_feat = self.norm(refined_local + local_tokens)
        refined_feat = self.mlp(local_feat) + local_feat

        return refined_feat, attn_mask
