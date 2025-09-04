import torch
import torch.nn.functional as F
def fuse_three_by_confidence(pred_fine, pred_macro, pred_attr, prototypes,
                             tau: float = 1.5,
                             beta: float = 0.7,
                             eps: float = 1e-6):
    logits_fine  = pred_fine  @ prototypes.t()
    logits_macro = pred_macro @ prototypes.t()
    logits_attr  = pred_attr  @ prototypes.t()
    p_fine  = F.softmax(logits_fine  / tau, dim=-1)
    p_macro = F.softmax(logits_macro / tau, dim=-1)
    p_attr  = F.softmax(logits_attr  / tau, dim=-1)
    H_fine  = -(p_fine  * (p_fine  + eps).log()).sum(-1)
    H_macro = -(p_macro * (p_macro + eps).log()).sum(-1)
    H_attr  = -(p_attr  * (p_attr  + eps).log()).sum(-1)
    C_fine  = 1.0 / H_fine.clamp(min=eps)
    C_macro = 1.0 / H_macro.clamp(min=eps)
    C_attr  = 1.0 / H_attr.clamp(min=eps)
    conf = torch.stack([C_fine, C_macro, C_attr], dim=1)
    w    = F.softmax(conf / beta, dim=1)
    w_fine, w_macro, w_attr = w[:, 0], w[:, 1], w[:, 2]
    fused = (w_fine.unsqueeze(1) * pred_fine +
             w_macro.unsqueeze(1) * pred_macro +
             w_attr.unsqueeze(1)  * pred_attr)
    return fused, (w_fine, w_macro, w_attr)