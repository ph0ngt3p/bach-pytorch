import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import tables
import numpy as np
from augmentations import rand_similarity_trans
from sklearn.model_selection import train_test_split

import cv2
import matplotlib.pyplot as plt


class MyTrainDataset(Dataset):
    def __init__(self, file, root_dir, train=True, transform=None):
        if os.path.isfile(file):
            self.train_file = np.load(file)
            self.train = train
            self.X = self.train_file['X_train']
            self.y = self.train_file['y_train']
            X_t, X_v, y_t, y_v = train_test_split(self.X, self.y, test_size=0.08, random_state=42, stratify=self.y)
            if self.train:
                self.X_train = X_t
                self.y_train = y_t
            else:
                self.X_val = X_v
                self.y_val = y_v
            self.root_dir = root_dir
            self.transform = transform
        else:
            print('Data path is not available!')
            exit(1)

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_val)

    def __getitem__(self, idx):
        if self.train:
            image = self.X_train[idx]
            label = self.y_train[idx]
        else:
            image = self.X_val[idx]
            label = self.y_val[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class MyTestDataset(Dataset):
    def __init__(self, file, root_dir, transform=None):
        if os.path.isfile(file):
            self.test_file = np.load(file)
            self.X_test = self.test_file['X_test']
            self.y_test = self.test_file['y_test']
            self.root_dir = root_dir
            self.transform = transform
        else:
            print('Data path is not available!')
            exit(1)

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, idx):
        image = self.X_test[idx]
        image_id = self.y_test[idx]

        if self.transform:
            image = self.transform(image)

        return image, image_id


def gen_outputline(label, preds):
    idx = str(label)
    return idx + ',' + str(preds)[1:-1].replace(',', '') + '\n'


def net_frozen(args, model):
    print('********************************************************')
    model.frozen_until(args.frozen_until)
    init_lr = args.lr
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                lr=init_lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                lr=init_lr,  weight_decay=args.weight_decay)
    print('********************************************************')
    return model, optimizer


def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
        else Variable(X)


def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return model


def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model


def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
        else Variable(X)


def show_dataset(dataset, n=6):
    img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                    for i in range(len(dataset))))
    plt.imshow(img)
    plt.axis('off')


def test_aug_train():
    dset = MyTrainDataset('data/train_val.npz', root_dir='./data')
    dset2 = MyTrainDataset('data/train_val.npz', train=False, root_dir='./data')

    print(len(dset))
    print(dset.X_train.shape)
    print(len(dset.y_train))
    print(dset.y_train)

    import collections
    print(collections.Counter(dset.y_train))
    print(collections.Counter(dset2.y_val))

    img = dset.X_train[1, :, :, :]
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# test_aug_train()
