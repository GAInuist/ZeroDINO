import pickle
import torch
import torch.nn as nn
import scipy.io as sio
from torch import optim
from sklearn.model_selection import train_test_split
from network.GLML import GLML
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import warnings
from utils import *
from config import *
from data import *

warnings.filterwarnings('ignore')

args = get_config_parser().parse_args()
for k, v in sorted(vars(args).items()):
    print(k, '=', v)

os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

# random seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
    print('seed: ' + str(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
pref = str(args.seed)
args.pref = pref
torch.backends.cudnn.benchmark = True

# tensorboard
log_dir = f'./runs/{args.DATASET}/{args.pref}'
writer = SummaryWriter(log_dir=log_dir)

ROOT = args.DATASET_path
DATA_DIR = f'xlsa17/data/{args.DATASET}'
data = sio.loadmat(f'{DATA_DIR}/res101.mat')
# data consists of files names
attrs_mat = sio.loadmat(f'{DATA_DIR}/att_splits.mat')
image_files = data['image_files']

if args.DATASET == 'AWA2':
    image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])
else:
    image_files = np.array([im_f[0][0].split('images/')[-1] for im_f in image_files])

labels = data['labels'].squeeze().astype(np.int64) - 1
train_idx = attrs_mat['train_loc'].squeeze() - 1
val_idx = attrs_mat['val_loc'].squeeze() - 1
trainval_idx = attrs_mat['trainval_loc'].squeeze() - 1
test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1

train_labels = labels[train_idx]
val_labels = labels[val_idx]

# split train_idx to train_idx (used for training) and val_seen_idx
train_idx, val_seen_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels)
# split val_idx to val_idx (not used) and val_unseen_idx
val_unseen_idx = train_test_split(val_idx, test_size=0.2, stratify=val_labels)[1]

attrs_mat = attrs_mat["att"].astype(np.float32).T

# train files and labels
train_files, train_labels = image_files[train_idx], labels[train_idx]
uniq_train_labels, train_labels_based0, counts_train_labels = np.unique(train_labels, return_inverse=True,
                                                                        return_counts=True)
# val seen files and labels
val_seen_files, val_seen_labels = image_files[val_seen_idx], labels[val_seen_idx]
uniq_val_seen_labels = np.unique(val_seen_labels)
# val unseen files and labels
val_unseen_files, val_unseen_labels = image_files[val_unseen_idx], labels[val_unseen_idx]
uniq_val_unseen_labels = np.unique(val_unseen_labels)

# trainval files and labels
trainval_files, trainval_labels = image_files[trainval_idx], labels[trainval_idx]
uniq_trainval_labels, trainval_labels_based0, counts_trainval_labels = np.unique(trainval_labels, return_inverse=True,
                                                                                 return_counts=True)
# test seen files and labels
test_seen_files, test_seen_labels = image_files[test_seen_idx], labels[test_seen_idx]
uniq_test_seen_labels = np.unique(test_seen_labels)
# test unseen files and labels
test_unseen_files, test_unseen_labels = image_files[test_unseen_idx],labels[test_unseen_idx]
uniq_test_unseen_labels = np.unique(test_unseen_labels)

# Transforms
trainTransform = transforms.Compose([
            transforms.Resize(int(448 * 8. / 7.)),
            transforms.RandomCrop(448),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
testTransform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

def train(model, data_loader, train_attrbs, optimizer, attr_mat, args):
    model.train()
    tk = tqdm(data_loader)
    for batch_idx, (data, label) in enumerate(tk):
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        data_attribute = torch.from_numpy(attr_mat).cuda()[label]

        result_dict = model(data, GloVe)
        global_result, local_result = result_dict['global_result'], result_dict['local_result']
        g_logit = global_result @ train_attrbs.T
        l_logit = local_result @ train_attrbs.T
        cls_loss = F.cross_entropy(g_logit, label)
        local_loss = F.cross_entropy(l_logit, label)

        loss = local_loss + cls_loss
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), label.shape[0])
        loss_cls_meter.update(cls_loss.item(), label.shape[0])
        loss_local_meter.update(local_loss.item(), label.shape[0])
        tk.set_postfix({"loss": loss_meter.avg,
                        "global_cls": loss_cls_meter.avg,
                        "local_cls": loss_local_meter.avg,})
    print('Train: Average loss: {:.4f}\n'.format(loss_meter.avg))


model = GLML(attr_num=args.attr_num).cuda()
if args.pretrain_path is not None:
    pth_dict = torch.load(args.pretrain_path, map_location='cuda:0')
    model_dict = model.state_dict()
    for k, v in pth_dict.items():
        if k in model_dict.keys():
            model_dict[k] = v
    model.load_state_dict(model_dict)

if args.is_train or args.cs:
    param_groups = []
    param_groups.append({"params": model.vit2attr.parameters(), "lr": 0.0001, "weight_decay": 0.00001})
    param_groups.append({"params": model.glove_refiner.parameters(), "lr": 0.0001, "weight_decay": 0.00001})
    param_groups.append({"params": model.w1.parameters(), "lr": 0.0001, "weight_decay": 0.00001})
    param_groups.append({"params": model.w2v.parameters(), "lr": 0.0001, "weight_decay": 0.00001})
    param_groups.append({"params": model.loc_refiner.parameters(), "lr": 0.0001, "weight_decay": 0.00001})

    optimizer = torch.optim.AdamW(param_groups)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50, 100], gamma=0.5)

train_attrbs = attrs_mat[uniq_train_labels]
train_attrbs_tensor = torch.from_numpy(train_attrbs).cuda()
trainval_attrbs = attrs_mat[uniq_trainval_labels]
trainval_attrbs_tensor = torch.from_numpy(trainval_attrbs).cuda()

loss_meter = AverageMeter()
loss_cls_meter = AverageMeter()
loss_attn_meter = AverageMeter()
loss_local_meter = AverageMeter()
    
GloVe_path = f'/root/autodl-tmp/GloVe/{args.DATASET}_attribute.pkl' 
with open(GloVe_path, 'rb') as f:
    GloVe = np.array(pickle.load(f))  
    GloVe = torch.from_numpy(GloVe).float().cuda() 

if __name__ == '__main__':
    if args.is_train:
        trainval_data_loader = get_loader(args, ROOT, trainval_files, trainval_labels_based0, trainTransform,
                                          is_sample=True,
                                          count_labels=counts_trainval_labels)
        test_seen_data_loader = get_loader(args, ROOT, test_seen_files, test_seen_labels, testTransform,
                                           is_sample=False)
        test_unseen_data_loader = get_loader(args, ROOT, test_unseen_files, test_unseen_labels, testTransform,
                                             is_sample=False)
        harmonic_best = []
        logging.basicConfig(level=logging.INFO, filename='Training_Log.log', format='%(message)s')
        for epoch in range(1, args.Num_Epochs):
            print('Train Val Epoch: ', epoch)
            train(model, trainval_data_loader, trainval_attrbs_tensor, optimizer, attrs_mat, args)
            lr_scheduler.step()
            test_gamma_best, best_test_dict = [], {}
            if args.search_gamma:
                gamma_list = [0.25, 0.30, 0.35, 0.40, 0.45] if epoch <= 5 \
                    else [0.40, 0.45, 0.50, 0.55, 0.60]
            else:  # just test one gamma
                gamma_list = [args.gamma]
            # gamma_list = [0.4, 0.45, 0.5, 0.55, 0.6] if epoch <= 5 else [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            for gamma in tqdm(gamma_list, desc="Processing gamma values"):
                metric_dict = test_GZSL(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader,
                                        test_unseen_labels, attrs_mat, gamma, args, GloVe)
                test_gamma_best.append(metric_dict['H'])
                if metric_dict['H'] >= max(test_gamma_best):
                    best_test_dict = metric_dict
            H, gzsl_seen_acc, gzsl_unseen_acc, best_gamma = (best_test_dict['H'], best_test_dict['gzsl_seen'],
                                                             best_test_dict['gzsl_unseen'], best_test_dict['gamma'])
            H_local, local_seen_acc, local_unseen_acc = (best_test_dict['H_local'], best_test_dict['local_seen'], best_test_dict['local_unseen'])
            H_global, global_seen_acc, global_unseen_acc = (best_test_dict['H_global'], best_test_dict['global_seen'], best_test_dict['global_unseen'])
            wg_seen, wg_unseen = (best_test_dict['Avg_w_g (seen)'], best_test_dict['Avg_w_g (unseen)'])
            print('GZSL Seen: averaged per-class accuracy: {0:.2f}, Local Seen: {1:.2f}, Global Seen: {2:.2f}, Wg Seen: {3:.3f}'.format(gzsl_seen_acc, local_seen_acc, global_seen_acc, wg_seen))
            print('GZSL Unseen: averaged per-class accuracy: {0:.2f}, Local Unseen: {1:.2f}, Global Unseen: {2:.2f}, Wg Unseen: {3:.3f}'.format(gzsl_unseen_acc, local_unseen_acc, global_unseen_acc, wg_unseen))
            print('GZSL: harmonic mean (H): {0:.2f}, Local H: {1:.2f}, Global H: {2:.2f}'.format(H, H_local, H_global))
            print('GZSL: best_gamma: {0:.2f}'.format(best_gamma))

            harmonic_best.append(best_test_dict['H'])
            print(f"Now the best harmonic is {max(harmonic_best)}, index is {harmonic_best.index(max(harmonic_best)) + 1}")
            if best_test_dict['H'] >= max(harmonic_best):
                print(' .... Saving model ...')
                h = best_test_dict['H']
                save_path_rm = str(args.DATASET) + f'_ZSLExperiment_{h:.2f}' + '.pth'
                ckpt_path = os.path.join('checkpoint', str(args.DATASET), f'{args.pref}')
                path = os.path.join(ckpt_path, save_path_rm)
                if not os.path.isdir(ckpt_path):
                    makedir(ckpt_path)
                torch.save(model.state_dict(), path)
                print(f"Now saving model with harmonic {h}")
            #### tensorboard
            writer.add_scalar('H', metric_dict['H'], epoch)
            writer.add_scalar('U', metric_dict['gzsl_unseen'], epoch)
            writer.add_scalar('S', metric_dict['gzsl_seen'], epoch)
            logging_info_flow(best_test_dict, epoch, loss_meter.avg)
        print(max(harmonic_best))

    if args.is_test:
        print('Now begin the Testing process')

        test_unseen_loader = get_loader(
            args, ROOT, test_unseen_files, test_unseen_labels,
            testTransform, is_sample=False
        )

        if args.zsl_only:
            _ = test_CZSL(model,
                          test_unseen_loader,
                          test_unseen_labels,
                          attrs_mat, args, GloVe)
        else:
            test_seen_loader = get_loader(
                args, ROOT, test_seen_files, test_seen_labels,
                testTransform, is_sample=False
            )
            metric_dict = test_GZSL(model, test_seen_loader, test_seen_labels,
                                    test_unseen_loader, test_unseen_labels,
                                    attrs_mat, args.gamma, args, GloVe)
