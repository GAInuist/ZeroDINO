import torch
from .utils import _init_weights
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SGD(nn.Module):
    def __init__(self,
                 attr_num: int = 312,
                 dim: int = 768):
        super(SGD, self).__init__()
        self.attr_num = attr_num
        self.word_embedding = nn.Linear(300, dim)
        self.fine_learner = Semantic_Block(self.attr_num, dim)
        self.coarse_learner = Semantic_Block(self.attr_num, dim)
        self.adafusion = AdaFusion(attr_num=self.attr_num, dim=dim)

    def forward(self, x_macro, x_fine, w2v, topk_fine=128, topk_macro=128):
        w2v = self.word_embedding(w2v).expand(x_macro.size(0), -1, -1)
        x_attr_fine, fine_weights = self.fine_learner(w2v, x_fine, topk_fine)
        x_attr_macro, macro_weights = self.coarse_learner(w2v, x_macro, topk_macro)
        x_attr, alpha, beta = self.adafusion(w2v, x_attr_fine, x_attr_macro)
        return x_attr,  x_attr_fine, x_attr_macro, fine_weights, macro_weights, alpha, beta




Norm = nn.LayerNorm

def adjust_tau(epoch, total_epochs, attn_module):
    initial_tau = 1.3
    final_tau = 0.3
    current_tau = initial_tau - (initial_tau - final_tau) * (epoch / total_epochs)
    if not attn_module.tau_trainable:
        attn_module.tau.data.fill_(current_tau)

class AnyAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, attn_drop=0.2, proj_drop=0.2,
                 bias=True, tau=0.4999, tau_trainable=True):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=tau_trainable)
        self.tau_trainable = tau_trainable
        if not tau_trainable:
            self.tau.requires_grad = False 

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(_init_weights)

    def forward(self, q, k, v, topk=10):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        B, Nq, C = q.shape
        _, Nk, _ = k.shape
        _, Nv, _ = v.shape

        q = self.q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        K = min(topk, attn_logits.shape[-1])
        topk_values, topk_indices = torch.topk(attn_logits, k=K, dim=-1)
        topk_softmax = torch.softmax(topk_values / self.tau.clamp(min=1e-8), dim=-1)
        sparse_attn = torch.zeros_like(attn_logits)
        sparse_attn.scatter_(-1, topk_indices, topk_softmax)
        attn = self.attn_drop(sparse_attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, attn


class MLP(nn.Module):
    def __init__(self, dim, drop_rate=0.2, mlp_ratio=2):
        super(MLP, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(dim)

        self.apply(_init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Semantic_Block(nn.Module):
    def __init__(self, attr_num, dim=768):
        super(Semantic_Block, self).__init__()
        self.dim = dim
        self.attr_num = attr_num
        self.attn = AnyAttention(dim=self.dim, num_heads=4, bias=True, tau_trainable=True)
        self.mlp = MLP(dim=self.dim, drop_rate=0.2, mlp_ratio=2)
        self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, w2v, local_tokens, topk=10):
        refined_local, attn_mask = self.attn(w2v, local_tokens, local_tokens, topk=topk)
        local_feat = self.norm(refined_local + w2v)
        refined_feat = self.mlp(local_feat) + local_feat


        return refined_feat, attn_mask

class AdaFusion(nn.Module):
    def __init__(self, attr_num=312, dim=768, bias=False, drop_rate=0.1):
        super().__init__()
        self.alpha_net = nn.Sequential(
            nn.Linear(dim, dim // 2, bias=bias),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim // 2, dim // 4, bias=bias),
            nn.GELU()
        )
        self.alpha = nn.Linear(dim // 4, 1, bias=bias)
        
        self.beta_net = nn.Sequential(
            nn.Linear(attr_num, attr_num // 2, bias=bias), 
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(attr_num // 2, attr_num // 4, bias=bias),
            nn.GELU()
        )
        self.beta = nn.Linear(attr_num // 4, 1, bias=bias)
        self.apply(_init_weights)

    def forward(self, w2v, x_attr_fine, x_attr_macro, epsilon=1e-6):
        alpha_feat = self.alpha_net(w2v)
        alpha = self.alpha(alpha_feat)
        alpha = F.sigmoid(alpha)
        
        beta_in = w2v.transpose(1, 2)
        beta_feat = self.beta_net(beta_in)
        beta = self.beta(beta_feat)
        beta = beta.transpose(1, 2)
        
        fused = alpha * ((x_attr_fine - x_attr_fine.mean(dim=-1, keepdims=True)) / (x_attr_fine.var(dim=-1, keepdims=True) + epsilon)) + (1-alpha) * ((x_attr_macro - x_attr_macro.mean(dim=-1, keepdims=True)) / (x_attr_macro.var(dim=-1, keepdims=True) + epsilon)) + beta
        
        return fused, alpha, beta