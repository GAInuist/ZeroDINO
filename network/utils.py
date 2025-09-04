import torch.nn as nn
import math
import torch

def _init_weights(moudle):
    if isinstance(moudle, nn.Linear):
        nn.init.kaiming_uniform_(moudle.weight, a=math.sqrt(5))
        if moudle.bias is not None:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(moudle.weight)
            bound1 = 1 / math.sqrt(fan_in1)
            nn.init.uniform_(moudle.bias, -bound1, bound1)
    if isinstance(moudle, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.zeros_(moudle.bias)
        nn.init.ones_(moudle.weight)
        

class SemanticAttention(nn.Module):
    def __init__(self, dim=768, attn_drop=0.0, proj_drop=0.0, bias=True):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        self.scale = dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)
        self.v = nn.Linear(dim, dim, bias=bias)

        self.lambda_scale = nn.Parameter(torch.tensor(0.5))
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(_init_weights)

    def forward(self, w2v, x, kernel=None):
        if kernel == None:
            weight = dist2weight(w2v, x)
        else:
            weight = learnable_dist2weight(w2v, x, kernel=kernel)
        q = self.q(self.norm_q(w2v))
        k = self.k(self.norm_k(x))
        v = self.v(self.norm_v(x))

        attn = (1 - self.lambda_scale) * (q @ k.transpose(-2, -1)) * self.scale + self.lambda_scale * weight
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, attn
    
class Self_Block(nn.Module):
    def __init__(self, dim=768):
        super(Self_Block, self).__init__()
        self.dim = dim
        self.attn = SemanticAttention(dim=self.dim, bias=True)
        self.mlp = MLP(dim=self.dim, drop_rate=0.2, mlp_ratio=2)
        self.norm = nn.LayerNorm(dim)
        self.norm_post = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, w2v, x_attr, kernel):
        refined_local, _ = self.attn(w2v, x_attr, kernel=kernel)
        local_feat = self.norm(refined_local + x_attr)
        refined_feat = self.mlp(local_feat) + local_feat
        feat = self.norm_post(refined_feat)
        return feat
    
class LearnableKernel(torch.nn.Module):
    def __init__(self, init_gamma=1.0, kernel_type='rbf'):
        super().__init__()
        self.logit_gamma = torch.nn.Parameter(torch.tensor(np.log(init_gamma)))
        self.kernel_type = kernel_type
        
    def forward(self, dist):
        gamma = 0.1 + 9.9 * torch.sigmoid(self.logit_gamma)
        
        if self.kernel_type == 'rbf':
            return torch.exp(-gamma * dist**2)
        elif self.kernel_type == 'laplacian':
            return torch.exp(-gamma * dist)
        elif self.kernel_type == 'adaptive':
            rbf = torch.exp(-gamma * dist**2)
            inv = 1 / (1 + dist + 1e-8)
            return 0.7*rbf + 0.3*inv
        else:
            raise ValueError("Unknown kernel type")
    
def dist2weight(w2v, x, func=lambda x: torch.tanh(10 * x)):
    d = torch.cdist(w2v, x)
    if func is not None:
        d = func(d)
    w = d / d.max(dim=-1, keepdims=True)[0]
    w = w + torch.eye(d.shape[-1], device=d.device).unsqueeze(0).tile([d.shape[0], 1, 1])
    return w

def learnable_dist2weight(w2v, x, kernel=None, normalize='max'):
    dist = torch.cdist(x, w2v)
    if kernel is None:
        kernel = LearnableKernel()
    weights = kernel(dist)
   
    min_val = 1e-8
    weights = torch.clamp(weights, min=min_val)
    
    if normalize == 'softmax':
        weights = torch.softmax(weights, dim=-1)
    elif normalize == 'max':
        max_vals = weights.max(dim=-1, keepdims=True)[0]
        weights = weights / torch.where(max_vals < min_val, 
                                      torch.ones_like(max_vals), 
                                      max_vals)
    identity = torch.eye(weights.shape[-1], device=weights.device).unsqueeze(0)
    weights = weights + identity
    
    return weights