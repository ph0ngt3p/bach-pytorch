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

import cv2
import matplotlib.pyplot as plt


class MyTrainDataset(Dataset):
    def __init__(self, hdf5_file, root_dir, train, aug_multiplier=5, transform=None):
        if os.path.isfile(hdf5_file):
            self.hdf5_file = tables.open_file(hdf5_file, mode='r')
            self.train = train
            if train:
                self.X_train = self.hdf5_file.root.X_train
                self.Y_train = self.hdf5_file.root.Y_train
            else:
                self.X_val = self.hdf5_file.root.X_val
                self.Y_val = self.hdf5_file.root.Y_val
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
            label = self.Y_train[idx]
        else:
            image = self.X_val[idx]
            label = self.Y_val[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class MyTestDataset(Dataset):
    def __init__(self, hdf5_file, root_dir, transform=None):
        if os.path.isfile(hdf5_file):
            self.hdf5_file = tables.open_file(hdf5_file, mode='r')
            self.X_test = self.hdf5_file.root.X_test
            self.Y_test = self.hdf5_file.root.Y_test
            self.root_dir = root_dir
            self.transform = transform
        else:
            print('Data path is not available!')
            exit(1)

    def __len__(self):
        return len(self.X_test)

    def __getitem__(self, idx):
        image = self.X_test[idx]
        image_id = self.Y_test[idx]

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


def test_aug_train():
    dset = MyTrainDataset('data/train_val_data.hdf5', root_dir='./data', train=True)
    print(len(dset))
    print(len(dset.Y_train))
    print(dset.Y_train)

    img = dset.X_train[-555, :, :, 0]
    print(dset.Y_train[-555])
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.show()


# test_aug_train()
