import os
import errno
import logging
import numpy as np
import torch
from scipy.special import softmax
from torchvision.transforms import transforms
import torch.nn.functional as F
from .utils_confidence import fuse_by_confidence

class AverageMeter(object):
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
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



def logging_info_flow(metric_dict, epoch, loss_item):
    H, U, S = metric_dict['H'], metric_dict['gzsl_unseen'], metric_dict['gzsl_seen']
    logging.info(f'Num_epoch:{epoch}\nLoss: {loss_item}  H:{H},  U:{U},  S:{S}\n')


def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        if np.sum(idx) == 0:
            acc_per_class[i] = 0
        else:
            acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)


def get_reprs(model, data_loader, args, w2v, attrs_mat):
    model.eval()
    attr_tensor = torch.from_numpy(attrs_mat).float().cuda()   # ← 提前转 Tensor
    global_reprs, local_reprs, reprs_fused, wg_all = [], [], [], []

    for data, _ in data_loader:
        data = data.cuda()
        with torch.no_grad():
            out = model(data, w2v)
            g_feat = out['global_result']
            l_feat = out['local_result']
            # Module - CEAF (refused)
            fused, w_g = fuse_by_confidence(g_feat, l_feat, attr_tensor)  # w_g: [B]

        global_reprs.append(g_feat.cpu().numpy())
        local_reprs.append(l_feat.cpu().numpy())
        reprs_fused.append(fused.cpu().numpy())
        wg_all.append(w_g.cpu().numpy())

    reprs_fused  = np.concatenate(reprs_fused,  0)
    local_reprs  = np.concatenate(local_reprs,  0)
    global_reprs = np.concatenate(global_reprs, 0)
    wg_mean      = np.concatenate(wg_all, 0).mean()

    return reprs_fused, local_reprs, global_reprs, wg_mean

def test_GZSL(model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels, attrs_mat, gamma,
              args, w2v):
    uniq_test_seen_labels = np.unique(test_seen_labels)
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    # Representation
    with torch.no_grad():
        seen_reprs_eve, seen_reprs_local, seen_reprs_global, wg_seen = get_reprs(model, test_seen_loader, args=args, w2v=w2v, attrs_mat=attrs_mat)
        unseen_reprs_eve, unseen_reprs_local, unseen_reprs_global, wg_unseen = get_reprs(model, test_unseen_loader, args=args, w2v=w2v, attrs_mat=attrs_mat)

    # Calibrated stacking
    Cs_mat = np.zeros(attrs_mat.shape[0])
    Cs_mat[uniq_test_seen_labels] = gamma

    # GZSL
    # seen classes every
    gzsl_seen_sim = softmax(seen_reprs_eve @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels = np.argmax(gzsl_seen_sim, axis=1)
    gzsl_seen_acc = compute_accuracy(gzsl_seen_predict_labels, test_seen_labels, uniq_test_seen_labels)
    gzsl_unseen_sim = softmax(unseen_reprs_eve @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels = np.argmax(gzsl_unseen_sim, axis=1)
    gzsl_unseen_acc = compute_accuracy(gzsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)
    H_eve = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc)

    gzsl_seen_sim_local = softmax(seen_reprs_local @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels_local = np.argmax(gzsl_seen_sim_local, axis=1)
    gzsl_seen_acc_local = compute_accuracy(gzsl_seen_predict_labels_local, test_seen_labels, uniq_test_seen_labels)
    gzsl_unseen_sim_local = softmax(unseen_reprs_local @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels_local = np.argmax(gzsl_unseen_sim_local, axis=1)
    gzsl_unseen_acc_local = compute_accuracy(gzsl_unseen_predict_labels_local, test_unseen_labels, uniq_test_unseen_labels)
    H_local = 2 * gzsl_unseen_acc_local * gzsl_seen_acc_local / (gzsl_unseen_acc_local + gzsl_seen_acc_local)

    gzsl_seen_sim_global = softmax(seen_reprs_global @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels_global = np.argmax(gzsl_seen_sim_global, axis=1)
    gzsl_seen_acc_global = compute_accuracy(gzsl_seen_predict_labels_global, test_seen_labels, uniq_test_seen_labels)
    gzsl_unseen_sim_global = softmax(unseen_reprs_global @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels_global = np.argmax(gzsl_unseen_sim_global, axis=1)
    gzsl_unseen_acc_global = compute_accuracy(gzsl_unseen_predict_labels_global, test_unseen_labels,
                                             uniq_test_unseen_labels)
    H_global = 2 * gzsl_unseen_acc_global * gzsl_seen_acc_global / (gzsl_unseen_acc_global + gzsl_seen_acc_global)

    package = {'H': H_eve * 100, 'gzsl_unseen': gzsl_unseen_acc * 100,
               'gzsl_seen': gzsl_seen_acc * 100, "gamma": gamma,
               'H_local': H_local * 100, 'local_unseen':gzsl_unseen_acc_local * 100, 'local_seen':gzsl_seen_acc_local * 100,
               'H_global': H_global * 100, 'global_unseen':gzsl_unseen_acc_global * 100, 'global_seen':gzsl_seen_acc_global * 100,
               'Avg_w_g (seen)': wg_seen, 'Avg_w_g (unseen)': wg_unseen}
    return package


def test_CZSL(model,test_unseen_loader,test_unseen_labels,attrs_mat,args,w2v):
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    with torch.no_grad():
        reprs_fused, _, _, wg_unseen = get_reprs(
            model, test_unseen_loader, args=args,
            w2v=w2v, attrs_mat=attrs_mat
        )

    zsl_unseen_sim   = softmax(reprs_fused @ attrs_mat[uniq_test_unseen_labels].T, axis=1)
    pred_idx         = np.argmax(zsl_unseen_sim, axis=1)
    pred_labels      = uniq_test_unseen_labels[pred_idx]

    zsl_unseen_acc   = compute_accuracy(pred_labels, test_unseen_labels,
                                        uniq_test_unseen_labels)

    print(f'CZSL (unseen only) accuracy: {zsl_unseen_acc*100:.2f}')
    print(f'Avg w_g (unseen): {wg_unseen:.3f}')

    return {'zsl_unseen': zsl_unseen_acc*100,
            'Avg_w_g (unseen)': wg_unseen}