import random
from network.Semantic_Guided_Decoder import adjust_tau, AnyAttention
from sovler import configure_optimizer
import wandb
import warnings
import json
import argparse
import torch.nn.functional as F
from types import SimpleNamespace
from network.Local_Prediction import DINOMulti
from utils import *
from three_eval_utils import *
from data import *

warnings.filterwarnings('ignore')
os.environ["WANDB_MODE"]="offline"


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/CUB_train.json', type=str, help='Path to config JSON file')
    cmd_args = parser.parse_args()
    with open(cmd_args.config, 'r') as f:
        config_dict = json.load(f)
    config = SimpleNamespace(**config_dict)
    config.pref = str(config.seed)
    return config

def setup_seed(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    args.seed = args.seed or random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    args.pref = str(args.seed)

def main(args):
    data, attrs_mat, image_files, ROOT = load_data(args)
    trainval_idx, test_seen_idx, test_unseen_idx = process_splits(data, attrs_mat, split_func='trainval')
    attrs_mat = attrs_mat["att"].astype(np.float32).T
    transforms_dict = create_transforms()
    loaders, attrbs_tensors, test_labels = create_data_loaders(args, data, transforms_dict, image_files, 
                                                  trainval_idx, test_seen_idx, test_unseen_idx, attrs_mat, ROOT)
    GloVe = load_glove(args)

    model = DINOMulti(attr_num=args.attr_num).cuda()
    load_pretrained(model, args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters     : {total_params / 1e6:.2f}M")
    print(f"Trainable Parameters : {trainable_params / 1e6:.2f}M")

    # Initialize WandB
    if args.is_train:
        wandb.init(project="ZSL_DINO", config=args, name=f"{args.DATASET}_{args.pref}")
        wandb.config.update(args)

    if args.is_train:
        optimizer, lr_scheduler = configure_optimizer(model)
        train_loop(model, loaders, attrbs_tensors, test_labels, optimizer, lr_scheduler, GloVe, args)
    else:
        print("=" * 40 + " Begin Testing " + "=" * 40)
        eval_loop(model, loaders, attrbs_tensors, test_labels, GloVe, args)
        print("=" * 42 + " Complete " + "=" * 43)


def train_loop(model, loaders, attrbs_tensors, test_labels, optimizer, lr_scheduler, GloVe, args):
    meters = {name: AverageMeter() for name in [
        'loss', 'local_loss', 'x_fine_loss', 'x_macro_loss', 'kl_loss']}
    harmonic_best = []
    logging.basicConfig(level=logging.INFO, filename=f'log/Training_Log_local_{args.pref}.log', format='%(message)s')

    for epoch in range(1, args.Num_Epochs + 1):
        print(f'Train Epoch: {epoch}')
        local_main(model, loaders['trainval'], attrbs_tensors['trainval_attrbs'], optimizer, GloVe, meters, epoch, args.Num_Epochs)
        lr_scheduler.step()

        if args.search_gamma:
            if epoch <= 5:
                gamma_list = [0.40, 0.45, 0.55]  # 0.5, 0.6, 0.7
            elif epoch <= 10 and epoch >= 5:
                gamma_list = [0.50, 0.55, 0.60]  # 0.6, 0.7, 0.8
            else:
                gamma_list = [0.55, 0.60, 0.65]


        else:
            gamma_list = [args.gamma]
        best_metrics = evaluate(model, loaders['test_seen'], loaders['test_unseen'], attrbs_tensors['all'], test_labels, gamma_list, GloVe, args)

        log_metrics(epoch, best_metrics, meters)
        harmonic_best.append(best_metrics['H'])

        if best_metrics['H'] >= max(harmonic_best):
            save_model(model, args, best_metrics['H'])

    print(f'Best Harmonic Mean: {max(harmonic_best):.2f}')


def local_main(model, loader, attrs_tensor, optimizer, GloVe, meters, epoch=None, total_epochs=None):
    model.train()
    if epoch is not None and total_epochs is not None:
        for module in model.modules():
            if isinstance(module, AnyAttention):
                adjust_tau(epoch, total_epochs, module)

    for data, label in tqdm(loader, desc="Training"):
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data, GloVe)
        data_attr = attrs_tensor[label]

        losses = calculate_losses(output, label, attrs_tensor, data_attr, epoch)
        optimizer.step()
        for name, value in losses.items():
            meters[name].update(value.item(), label.size(0))
    log_str = " | ".join([f"{name}: {meter.avg:.4f}" for name, meter in meters.items()])
    print(f"Train: {log_str}")

def get_temperature(epoch, T_max=1, T_min=0.5, T_decay=10):
    temperature = T_min + (T_max - T_min) * np.exp(-epoch / T_decay)
    return temperature


def calculate_losses(output, labels, attrs_tensor, data_attr, epoch):
    local_out = output['fine_result']
    x_fine_out = output['x_fine_result']
    x_macro_out = output['x_macro_result']
    # Classification losses
    cls_losses = {
        'local': F.cross_entropy(local_out @ attrs_tensor.T, labels),
        'x_fine': F.cross_entropy(x_fine_out @ attrs_tensor.T, labels),
        'x_macro': F.cross_entropy(x_macro_out @ attrs_tensor.T, labels),
    }
    
    pool_fine, pool_macro = output['pool_fine'], output['pool_macro']
    temperature = get_temperature(epoch)
    p = F.softmax(pool_fine / temperature, dim=-1).detach()
    q = F.softmax(pool_macro / temperature, dim=-1)
    m = 0.5 * (p + q)
    kl_p_m = 0.5 * F.kl_div(p.log(), m, reduction='batchmean')
    kl_q_m = 0.5 * F.kl_div(q.log(), m, reduction='batchmean')
    js_loss = (kl_p_m + kl_q_m) * (temperature ** 2)

    total_loss = cls_losses['local'] + 5 * js_loss + cls_losses['x_fine'] + cls_losses['x_macro']
    total_loss.backward()
    # Total losses
    return {
        'local_loss': cls_losses['local'],
        'x_fine_loss': cls_losses['x_fine'],
        'x_macro_loss': cls_losses['x_macro'],
        'kl_loss': js_loss,
        'loss': total_loss
    }


def log_metrics(epoch, metrics, meters):
    # WandB logging
    wandb_log = {
        'epoch': epoch,
        'train/loss': meters['loss'].avg,
        'train/local_loss': meters['local_loss'].avg,
        'train/x_fine_loss': meters['x_fine_loss'].avg,
        'train/x_macro_loss': meters['x_macro_loss'].avg,
        'train/kl_loss': meters['kl_loss'].avg,
        'test/H': metrics['H'],
        'test/seen_acc': metrics['gzsl_seen'],
        'test/unseen_acc': metrics['gzsl_unseen'],
        'test/gamma': metrics['gamma']
    }
    wandb.log(wandb_log)
    logging_info_flow(metrics, epoch, meters['loss'].avg)

def eval_loop(model, loaders, attrbs_tensors, test_labels, GloVe, args):
    best_metrics = evaluate(model, loaders['test_seen'], loaders['test_unseen'], attrbs_tensors['all'], test_labels, [args.gamma], GloVe, args)
    
if __name__ == '__main__':
    args = get_config()
    print("=" * 40 + " Configuration " + "=" * 40)
    for k, v in vars(args).items():
        print(f"{k:25} = {v}")
    print("=" * 94)
    setup_seed(args)
    main(args)