from torchvision import transforms
from sklearn.model_selection import train_test_split
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class DataLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root = root
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img_pil = Image.open(os.path.join(self.root, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)

def load_data(args):
    ROOT = args.DATASET_path
    DATA_DIR = f'{args.xlsa17_path}/{args.DATASET}'
    data = sio.loadmat(f'{DATA_DIR}/res101.mat')
    attrs_mat = sio.loadmat(f'{DATA_DIR}/att_splits.mat')
    image_files = data['image_files']
    if args.DATASET == 'AWA2':
        image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])
    else:
        image_files = np.array([im_f[0][0].split('images/')[-1] for im_f in image_files])
    return data, attrs_mat, image_files, ROOT


def process_splits(data, attrs_mat, split_func='trainval'):
    labels = data['labels'].squeeze().astype(np.int64) - 1
    if split_func == 'trainval':
        trainval_idx = attrs_mat['trainval_loc'].squeeze() - 1
        test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
        test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1
        return trainval_idx, test_seen_idx, test_unseen_idx
    else:
        train_idx = attrs_mat['train_loc'].squeeze() - 1
        val_idx = attrs_mat['val_loc'].squeeze() - 1
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_idx, val_seen_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels)
        val_unseen_idx = train_test_split(val_idx, test_size=0.2, stratify=val_labels)[1]
        return train_idx, val_seen_idx, val_unseen_idx


def create_transforms():
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return {
        'train': transforms.Compose([
            transforms.Resize(int(448 * 8 / 7)),
            transforms.RandomCrop(448),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            normalize
        ])
    }

def create_data_loaders(args, data, transforms, image_files, trainval_idx, test_seen_idx, test_unseen_idx, attrs_mat, ROOT):
    labels = data['labels'].squeeze().astype(np.int64) - 1
    # trainval files and labels
    trainval_files, trainval_labels = image_files[trainval_idx], labels[trainval_idx]
    uniq_trainval_labels, trainval_labels_based0, counts_trainval_labels = np.unique(trainval_labels,return_inverse=True,return_counts=True)
    # test seen files and labels
    test_seen_files, test_seen_labels = image_files[test_seen_idx], labels[test_seen_idx]
    # test unseen files and labels
    test_unseen_files, test_unseen_labels = image_files[test_unseen_idx], labels[test_unseen_idx]

    # Create data loaders
    trainval_data_loader = get_loader(args, ROOT, trainval_files, trainval_labels_based0, transforms['train'],
                                      is_sample=True, count_labels=counts_trainval_labels)
    test_seen_data_loader = get_loader(args, ROOT, test_seen_files, test_seen_labels, transforms['test'],
                                       is_sample=False)
    test_unseen_data_loader = get_loader(args, ROOT, test_unseen_files, test_unseen_labels, transforms['test'],
                                         is_sample=False)

    test_labels = {
        'seen': test_seen_labels,
        'unseen': test_unseen_labels
    }
    loaders = {
        'trainval': trainval_data_loader,
        'test_seen': test_seen_data_loader,
        'test_unseen': test_unseen_data_loader
    }
    attrbs = {
        'trainval_attrbs': torch.from_numpy(attrs_mat[uniq_trainval_labels]).cuda(),
        'all' : attrs_mat
    }

    return loaders, attrbs, test_labels


def get_loader(args, ROOT, files, labels_based0, transform, is_sample=False, count_labels=None):
    data = DataLoader(ROOT, files, labels_based0, transform=transform)
    if is_sample:
        weights_ = 1. / count_labels
        weights = weights_[labels_based0]
        train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=labels_based0.shape[0],
                                                               replacement=True)
        data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=train_sampler,
                                                  num_workers=args.num_workers)
        return data_loader
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.test_batch, shuffle=False,
                                              num_workers=args.num_workers)
    return data_loader