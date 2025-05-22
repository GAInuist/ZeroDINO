import timm
import torch
from torch import nn

class DINO_ViT(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=pretrained,
            img_size=448,
            num_classes=0,
            dynamic_img_size=True)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        features = self.model.forward_features(x)
        if isinstance(features, dict):
            cls_token = features["x_norm_clstoken"]
            img_tokens = features["x_norm_patchtokens"]
        else:
            cls_token = features[:, 0]
            img_tokens = features[:, 1:]
        return cls_token, img_tokens


