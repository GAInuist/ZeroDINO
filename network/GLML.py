import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .ViT_Base import DINO_ViT
from .Local_Refiner import Local_Refiner
from .GloVe_Refiner import GloVe_Refiner


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound)

    if isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class GLML(nn.Module):
    def __init__(self, attr_num, dim=768):
        super(GLML, self).__init__()
        self.attr_num = attr_num
        self.dim = dim
        self.vit = DINO_ViT()
        self.vit2attr = nn.Linear(dim, self.attr_num)
        self.w1 = nn.Linear(dim, self.attr_num)
        self.w2v = nn.Linear(300, self.dim)
        self.loc_refiner = Local_Refiner(self.attr_num, self.dim)
        self.glove_refiner = GloVe_Refiner(self.attr_num, self.dim, momentum=0.8)
        self.vit2attr.apply(_init_weights)
        self.w1.apply(_init_weights)
        self.w2v.apply(_init_weights)
        self.loc_refiner.apply(_init_weights)
        self.glove_refiner.apply(_init_weights)

    def forward(self, x, w2v):
        B = x.shape[0]
        cls_token, local_tokens = self.vit(x)
        global_result = self.vit2attr(cls_token)
        w2v = self.w2v(w2v)
        refined_w2v = self.glove_refiner(local_tokens, w2v)
        refined_w2v = refined_w2v.expand(B, -1, -1)
        refined_local, attn_mask = self.loc_refiner(local_tokens, refined_w2v)
        region_weight = torch.einsum('bre, ae -> bra', refined_local, self.w1.weight)
        region_weight = F.softmax(region_weight, dim=1)
        attr_feature = torch.einsum('bre, bra -> bae', refined_local, region_weight)
        local_result = torch.einsum('bae, ae -> ba', attr_feature, self.w1.weight)
        package = {'global_result': global_result, 'local_result': local_result}
        return package
