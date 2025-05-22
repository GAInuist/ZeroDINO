# method - CEAF

import torch, torch.nn.functional as F

def fuse_by_confidence(g_feat, l_feat, prototypes,
                       tau: float = 1.5,
                       eps: float = 1e-6,
                       w_clip: tuple = (0.05, 0.95)):
    logits_g = g_feat @ prototypes.t()
    logits_l = l_feat @ prototypes.t()
    p_g = F.softmax((logits_g / tau))
    p_l = F.softmax((logits_l / tau))
    H_g = -(p_g * (p_g + eps).log()).sum(-1)
    H_l = -(p_l * (p_l + eps).log()).sum(-1)
    C_g = 1.0 / H_g
    C_l = 1.0 / H_l
    C_g = C_g.clamp(max=1e4)
    C_l = C_l.clamp(max=1e4)
    w_g = (C_g / (C_g + C_l + eps)).clamp(*w_clip)
    w_l = 1.0 - w_g
    fused = w_g.unsqueeze(1) * g_feat + w_l.unsqueeze(1) * l_feat

    return fused, w_g
