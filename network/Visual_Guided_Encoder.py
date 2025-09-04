import timm
import math
import torch
import torch.nn as nn


class VGE(nn.Module):
    def __init__(self,
                 pretrained=True,
                 dim: int = 768,
                 n_head: int = 8,
                 depth: int = 1,
                 patch_size: int = 32,
                 freeze_backbone: bool = True):
        super(VGE,self).__init__()
        self.patch_size = patch_size
        self.model = timm.create_model(
            "vit_base_patch14_dinov2", pretrained=pretrained,
            img_size=448, num_classes=0, dynamic_img_size=True)
        self.downsample_macro = AvgPoolDown()


        self.cross_fine = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, dropout=0.2,
            bias=True, batch_first=True)
        self.norm_fine = nn.LayerNorm(dim)
        self.pool_fine = nn.AdaptiveAvgPool1d(1)

        self.cross_macro = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, dropout=0.2,
            bias=True, batch_first=True)
        self.norm_macro = nn.LayerNorm(dim)
        self.pool_macro = nn.AdaptiveAvgPool1d(1)

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        B = x.size(0)
        features = self.model.forward_features(x)
        x_global = features[:, 0].detach()
        x_fine = features[:, 1:].detach()

        # ---------- Coarse分支 ----------
        x_macro = self.model.patch_embed(x)
        x_macro = x_macro.reshape(B, self.patch_size ** 2, 768)
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x_macro = torch.cat((cls_token, x_macro), dim=1)
        x_macro = self.model.pos_drop(x_macro + self.model.pos_embed)

        for i in range(len(self.model.blocks) - 1):
            x_macro = self.model.blocks[i](x_macro)
        x_macro_nocls = x_macro[:, 1:]
        H = W = int(math.sqrt(x_macro_nocls.size(1)))
        x_macro = self.downsample_macro(x_macro_nocls, H, W)
        x_fine_cross, _ = self.cross_fine(x_fine, x_macro, x_macro)
        x_fine = self.norm_fine(x_fine + x_fine_cross)
        pool_fine = self.pool_fine(x_fine.permute(0, 2, 1)).permute(0, 2, 1)
        x_macro_cross, _ = self.cross_macro(x_macro, x_fine, x_fine)
        x_macro = self.norm_macro(x_macro + x_macro_cross)
        pool_macro = self.pool_macro(x_macro.permute(0, 2, 1)).permute(0, 2, 1)

        return x_macro, x_fine, x_global, pool_fine, pool_macro


class AvgPoolDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

    def forward(self, x_seq, H, W):
        B, N, C = x_seq.shape
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        x = self.pool1(x)
        x = self.pool2(x)
        Hd, Wd = x.shape[2:]
        x = x.reshape(B, C, Hd * Wd).transpose(1, 2)
        return x
