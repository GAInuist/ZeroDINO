from torch import optim

def configure_optimizer(model):
    optimizer = optim.Adam(model.parameters())

    params = [
        # local learning
        {"params": model.VGE.downsample_macro.parameters(), "lr": 1e-6, "weight_decay": 1e-5},
        {"params": model.VGE.cross_fine.parameters(), "lr": 1e-6, "weight_decay": 1e-5},
        {"params": model.VGE.norm_fine.parameters(), "lr": 1e-6, "weight_decay": 1e-5},
        {"params": model.VGE.pool_fine.parameters(), "lr": 1e-6, "weight_decay": 1e-5},
        {"params": model.VGE.cross_macro.parameters(), "lr": 1e-6, "weight_decay": 1e-5},
        {"params": model.VGE.norm_macro.parameters(), "lr": 1e-6, "weight_decay": 1e-5},
        {"params": model.VGE.pool_macro.parameters(), "lr": 1e-6, "weight_decay": 1e-5},

        {"params": model.SGD.parameters(), "lr": 1e-6, "weight_decay": 1e-5},
        {"params": model.fine_predictor.parameters(), "lr": 1e-6, "weight_decay": 1e-3},
        {"params": model.x_fine_predictor.parameters(), "lr": 1e-6, "weight_decay": 1e-3},
        {"params": model.x_macro_predictor.parameters(), "lr": 1e-6, "weight_decay": 1e-3},
    ]
    optimizer = optim.AdamW(params)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 50], gamma=0.5)
    return optimizer, lr_scheduler