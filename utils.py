import os
import errno
import logging
import torch
import pickle


class AverageMeter(object):
    """class for managing loss function values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def makedir(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_pretrained(model, args):
    if args.pretrain_path:
        state_dict = torch.load(args.pretrain_path, map_location='cuda:0')
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            if k in state_dict.keys():
                model_dict[k] = state_dict[k]
            else:
                continue
        model.load_state_dict(model_dict)

def logging_info_flow(metric_dict, epoch, loss_item):
    H = round(metric_dict['H'], 3)
    U = round(metric_dict['gzsl_unseen'], 3)
    S = round(metric_dict['gzsl_seen'], 3)
    C = round(metric_dict['czsl'], 3)
    gamma = round(metric_dict['gamma'], 3)
    loss_item = round(loss_item, 3)
    logging.info(f'Num_epoch:{epoch}\nLoss: {loss_item} | H:{H} | U:{U} | S:{S} | C:{C} | Gamma:{gamma}\n')

def load_glove(args):
    with open(f'./w2v/{args.DATASET}_attribute.pkl', 'rb') as f:
        return torch.from_numpy(pickle.load(f)).float().cuda()
    
def save_model(model, args, h_score):
    save_path = f'checkpoint/{args.DATASET}/{args.pref}/{args.DATASET}_NoS_o4_{h_score:.2f}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved model with H-score: {h_score:.2f}")

