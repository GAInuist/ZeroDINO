import torch
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from utils_confidence import *

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
    attr_tensor = torch.from_numpy(attrs_mat).float().cuda()
    global_reprs, local_reprs, reprs_fused = [], [], []
    x_fine_reprs, x_macro_reprs = [], []
    fusion_weights_fine, fusion_weights_macro, fusion_weights_local = [], [], []
    alpha_list, beta_list = [], []

    for data, _ in data_loader:
        data = data.cuda()
        with torch.no_grad():
            output = model(data, w2v)
            l_feat = output['fine_result']
            x_fine_feat = output['x_fine_result']
            x_macro_feat = output['x_macro_result']
            g_feat = torch.randn_like(l_feat).cuda()
            alpha = output['alpha']
            beta = output['beta']
            fused, weights = fuse_three_by_confidence(x_fine_feat, x_macro_feat, l_feat, attr_tensor)
            w_fine, w_macro, w_local = weights

        global_reprs.append(g_feat.cpu().numpy())
        local_reprs.append(l_feat.cpu().numpy())
        x_fine_reprs.append(x_fine_feat.cpu().numpy())
        x_macro_reprs.append(x_macro_feat.cpu().numpy())
        reprs_fused.append(fused.cpu().numpy())
        fusion_weights_fine.append(w_fine.cpu().numpy())
        fusion_weights_macro.append(w_macro.cpu().numpy())
        fusion_weights_local.append(w_local.cpu().numpy())
        alpha_list.append(alpha.cpu().numpy())
        beta_list.append(beta.cpu().numpy())

    reprs_fused = np.concatenate(reprs_fused, 0)
    local_reprs = np.concatenate(local_reprs, 0)
    global_reprs = np.concatenate(global_reprs, 0)
    x_fine_reprs = np.concatenate(x_fine_reprs, 0)
    x_macro_reprs = np.concatenate(x_macro_reprs, 0)
    fusion_weights_fine = np.concatenate(fusion_weights_fine, 0)
    fusion_weights_macro = np.concatenate(fusion_weights_macro, 0)
    fusion_weights_local = np.concatenate(fusion_weights_local, 0)
    alpha_list = np.concatenate(alpha_list, 0)
    beta_list = np.concatenate(beta_list, 0)

    return reprs_fused, local_reprs, global_reprs, x_fine_reprs, x_macro_reprs, fusion_weights_fine, fusion_weights_macro, fusion_weights_local, alpha_list, beta_list

def test_GZSL(model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels, attrs_mat, gamma,
              args, w2v):
    uniq_test_seen_labels = np.unique(test_seen_labels)
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    with torch.no_grad():
        seen_reprs_eve, seen_reprs_local, seen_reprs_global, seen_reprs_x_fine, seen_reprs_x_macro, seen_weights_fine, seen_weights_macro, seen_weights_local, seen_alpha_list, seen_beta_list = get_reprs(model, test_seen_loader,args=args, w2v=w2v,attrs_mat=attrs_mat)
        unseen_reprs_eve, unseen_reprs_local, unseen_reprs_global, unseen_reprs_x_fine, unseen_reprs_x_macro, unseen_weights_fine, unseen_weights_macro, unseen_weights_local, unseen_alpha_list, unseen_beta_list = get_reprs(model,test_unseen_loader,args=args, w2v=w2v,attrs_mat=attrs_mat)

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
    gzsl_unseen_acc_local = compute_accuracy(gzsl_unseen_predict_labels_local, test_unseen_labels,
                                             uniq_test_unseen_labels)
    H_local = 2 * gzsl_unseen_acc_local * gzsl_seen_acc_local / (gzsl_unseen_acc_local + gzsl_seen_acc_local)

    gzsl_seen_sim_global = softmax(seen_reprs_global @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels_global = np.argmax(gzsl_seen_sim_global, axis=1)
    gzsl_seen_acc_global = compute_accuracy(gzsl_seen_predict_labels_global, test_seen_labels, uniq_test_seen_labels)
    gzsl_unseen_sim_global = softmax(unseen_reprs_global @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels_global = np.argmax(gzsl_unseen_sim_global, axis=1)
    gzsl_unseen_acc_global = compute_accuracy(gzsl_unseen_predict_labels_global, test_unseen_labels,
                                              uniq_test_unseen_labels)
    H_global = 2 * gzsl_unseen_acc_global * gzsl_seen_acc_global / (gzsl_unseen_acc_global + gzsl_seen_acc_global)

    # Evaluate x_fine branch
    gzsl_seen_sim_x_fine = softmax(seen_reprs_x_fine @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels_x_fine = np.argmax(gzsl_seen_sim_x_fine, axis=1)
    gzsl_seen_acc_x_fine = compute_accuracy(gzsl_seen_predict_labels_x_fine, test_seen_labels, uniq_test_seen_labels)

    gzsl_unseen_sim_x_fine = softmax(unseen_reprs_x_fine @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels_x_fine = np.argmax(gzsl_unseen_sim_x_fine, axis=1)
    gzsl_unseen_acc_x_fine = compute_accuracy(gzsl_unseen_predict_labels_x_fine, test_unseen_labels,
                                              uniq_test_unseen_labels)
    H_x_fine = 2 * gzsl_unseen_acc_x_fine * gzsl_seen_acc_x_fine / (gzsl_unseen_acc_x_fine + gzsl_seen_acc_x_fine)

    # Evaluate x_macro branch
    gzsl_seen_sim_x_macro = softmax(seen_reprs_x_macro @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels_x_macro = np.argmax(gzsl_seen_sim_x_macro, axis=1)
    gzsl_seen_acc_x_macro = compute_accuracy(gzsl_seen_predict_labels_x_macro, test_seen_labels, uniq_test_seen_labels)

    gzsl_unseen_sim_x_macro = softmax(unseen_reprs_x_macro @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels_x_macro = np.argmax(gzsl_unseen_sim_x_macro, axis=1)
    gzsl_unseen_acc_x_macro = compute_accuracy(gzsl_unseen_predict_labels_x_macro, test_unseen_labels,
                                               uniq_test_unseen_labels)
    H_x_macro = 2 * gzsl_unseen_acc_x_macro * gzsl_seen_acc_x_macro / (gzsl_unseen_acc_x_macro + gzsl_seen_acc_x_macro)

    mean_fine = np.mean(np.concatenate([seen_weights_fine, unseen_weights_fine], axis=0))
    mean_macro = np.mean(np.concatenate([seen_weights_macro, unseen_weights_macro], axis=0))
    mean_local = np.mean(np.concatenate([seen_weights_local, unseen_weights_local], axis=0))

    mean_alpha = np.mean(np.concatenate([seen_alpha_list, unseen_alpha_list], axis=0))
    mean_beta = np.mean(np.concatenate([seen_beta_list, unseen_beta_list], axis=0))

    package = {
        'H': H_eve * 100, 'gzsl_unseen': gzsl_unseen_acc * 100, 'gzsl_seen': gzsl_seen_acc * 100, "gamma": gamma,
        'H_local': H_local * 100, 'local_unseen': gzsl_unseen_acc_local * 100, 'local_seen': gzsl_seen_acc_local * 100,
        'H_global': H_global * 100, 'global_unseen': gzsl_unseen_acc_global * 100,
        'global_seen': gzsl_seen_acc_global * 100,
        'H_x_fine': H_x_fine * 100, 'x_fine_unseen': gzsl_unseen_acc_x_fine * 100,
        'x_fine_seen': gzsl_seen_acc_x_fine * 100,
        'H_x_macro': H_x_macro * 100, 'x_macro_unseen': gzsl_unseen_acc_x_macro * 100,
        'x_macro_seen': gzsl_seen_acc_x_macro * 100,
        'mean_fine': mean_fine,
        'mean_macro': mean_macro,
        'mean_local': mean_local,
        'mean_alpha': mean_alpha,
        'mean_beta': mean_beta,
    }
    return package


def test_CZSL(model, test_unseen_loader, test_unseen_labels, attrs_mat, args, w2v):
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    with torch.no_grad():
        reprs_fused, _, _, _, _, _, _, _, _, _ = get_reprs(model, test_unseen_loader, args=args, w2v=w2v, attrs_mat=attrs_mat)

    zsl_unseen_sim = softmax(reprs_fused @ attrs_mat[uniq_test_unseen_labels].T, axis=1)
    pred_idx = np.argmax(zsl_unseen_sim, axis=1)
    pred_labels = uniq_test_unseen_labels[pred_idx]

    zsl_unseen_acc = compute_accuracy(pred_labels, test_unseen_labels,
                                      uniq_test_unseen_labels)

    return zsl_unseen_acc * 100


def evaluate(model, seen_loader, unseen_loader, attrs_tensor, test_labels, gamma_list, GloVe, args):
    test_gamma_best, best_test_dict = [], {}
    for gamma in tqdm(gamma_list, desc="Processing gamma values"):
        metric_dict = test_GZSL(model, seen_loader, test_labels['seen'], unseen_loader, test_labels['unseen'], attrs_tensor, gamma, args, GloVe)
        test_gamma_best.append(metric_dict['H'])
        if metric_dict['H'] >= max(test_gamma_best):
            best_test_dict = metric_dict
    H, gzsl_seen_acc, gzsl_unseen_acc, best_gamma = (best_test_dict['H'], best_test_dict['gzsl_seen'],
                                                     best_test_dict['gzsl_unseen'], best_test_dict['gamma'])
    H_local, local_seen_acc, local_unseen_acc = (
    best_test_dict['H_local'], best_test_dict['local_seen'], best_test_dict['local_unseen'])
    H_global, global_seen_acc, global_unseen_acc = (
    best_test_dict['H_global'], best_test_dict['global_seen'], best_test_dict['global_unseen'])
    H_x_fine, x_fine_seen_acc, x_fine_unseen_acc = (best_test_dict['H_x_fine'], best_test_dict['x_fine_seen'], best_test_dict['x_fine_unseen'])
    H_x_macro, x_macro_seen_acc, x_macro_unseen_acc = (best_test_dict['H_x_macro'], best_test_dict['x_macro_seen'], best_test_dict['x_macro_unseen'])
    mean_fine, mean_macro, mean_local = best_test_dict['mean_fine'], best_test_dict['mean_macro'], best_test_dict['mean_local']
    mean_alpha, mean_beta = best_test_dict['mean_alpha'], best_test_dict['mean_beta']
    czsl_acc = test_CZSL(model, unseen_loader, test_labels['unseen'], attrs_tensor, args, GloVe)
    best_test_dict['czsl'] = czsl_acc
    print('GZSL Seen: {0:.2f}, Local Seen: {1:.2f}, Global Seen: {2:.2f}, x_fine Seen: {3:.2f}, x_macro Seen: {4:.2f}'.format(gzsl_seen_acc, local_seen_acc, global_seen_acc, x_fine_seen_acc, x_macro_seen_acc))
    print('GZSL Unseen: {0:.2f}, Local Unseen: {1:.2f}, Global Unseen: {2:.2f}, x_fine Unseen: {3:.2f}, x_macro Unseen: {4:.2f}'.format(gzsl_unseen_acc, local_unseen_acc, global_unseen_acc, x_fine_unseen_acc, x_macro_unseen_acc))
    print('GZSL H: {0:.2f}, Local H: {1:.2f}, Global H: {2:.2f}, x_fine H: {3:.2f}, x_macro H: {4:.2f}'.format(H, H_local, H_global, H_x_fine, H_x_macro))
    print(f'Alpha Mean: {mean_alpha:.4f}, Beta Mean: {mean_beta:.4f}')
    print(f'Fusion Weights (averaged across dataset): Fine: {mean_fine:.4f} | Macro: {mean_macro:.4f} | Local: {mean_local:.4f}')
    print('GZSL best_gamma: {:.2f}'.format(best_gamma))
    print('CZSL Top-1 Acc: {:.2f}'.format(czsl_acc))
    return best_test_dict