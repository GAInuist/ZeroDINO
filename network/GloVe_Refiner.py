# Module - SPR

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GloVe_Refiner(nn.Module):
    def __init__(self, attr_num, feature_dim=768, momentum=0.8):
        super(GloVe_Refiner, self).__init__()
        self.attr_num = attr_num
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.mlp = MLP(in_features=self.feature_dim, hidden_features=self.feature_dim // 2)

    def forward(self, local_tokens, glove):
        m, d = glove.shape
        normalized_glove = F.normalize(glove, dim=-1)
        local_feat = local_tokens.reshape(-1, d)
        normalized_local_feat = F.normalize(local_feat, dim=-1)

        softmax_score_query, softmax_score_cache = self.get_score(normalized_glove, normalized_local_feat)
        _, top_idx = torch.max(softmax_score_query, dim=0)
        updated_glove = normalized_glove.clone().detach()

        for i in range(m):
            patch_idx = torch.nonzero(top_idx == i)
            a, _ = patch_idx.size()
            if a != 0:
                num = softmax_score_cache[i, patch_idx.squeeze(1)]
                denom = softmax_score_cache[i, :].max()
                w = num / (denom + 1e-9)
                feats = local_feat[patch_idx.squeeze(1), :]
                mean_new = (w.unsqueeze(1) * feats).sum(0)
                updated_glove[i] = (
                        self.momentum * normalized_glove[i] +
                        (1 - self.momentum) * mean_new
                )

        updated_glove = F.normalize(updated_glove, dim=-1)
        refined_glove = self.mlp(updated_glove + glove)

        return refined_glove

    def get_score(self, query, mem):
        score = query @ mem.t()
        score_query = F.softmax(score, dim=0)
        score_mem = F.softmax(score, dim=1)
        return score_query, score_mem

