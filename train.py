import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
import argparse
import torch.utils.data as utilsData
import torchvision.models as models
import numpy as np
import csv
import torchvision.transforms as transforms
import datetime
import time
import copy
from tqdm import tqdm
from logger import Logger

from model import *
from utils import *
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-2, type=float, help='Learning Rate')
parser.add_argument('--depth', default=18, choices=[18, 34, 50, 152, 161], type=int, help='depth of model')
parser.add_argument('--optim', default='sgd', choices=['adam', 'sgd'], type=str,
                    help='Using Adam or SGD optimizer for model')
parser.add_argument('--inspect', '-ins', action='store_true', help='Inspect saved model')
parser.add_argument('--interval', default=250, type=int, help='Number of epochs to train the model')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='Weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size - default: 256')
parser.add_argument('--train_from', default=1,
                    choices=[0, 1, 2],  # 0: from scratch, 1: from pretrained Resnet, 2: specific checkpoint in model_path
                    type=int,
                    help="training from beginning (1) or from the most recent ckpt (0)")
parser.add_argument('--check_after', default=1, type=int, help='Validate the model after how many epoch - default : 1')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--frozen_until', '-fu', type=int, default=8,
                    help="freeze until --frozen_util block")
args = parser.parse_args()
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Available device is:{}'.format(device))
print(torch.cuda.current_device)

start_epoch = 0
save_acc = 0
save_loss = 0

###########
print('DataLoader ....')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# input_size = 224
input_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

train_logger = Logger(os.path.join(LOG_DIR, 'train'))
val_logger = Logger(os.path.join(LOG_DIR, 'val'))


# cudnn.benchmark = True
def exp_lr_schedule(args, optimizer, epoch):
    # after epoch 100, not more learning rate decay
    init_lr = args.lr
    lr_decay_epoch = 50  # decay lr after each 10 epoch
    weight_decay = args.weight_decay
    lr = init_lr * (0.6 ** (min(epoch, 100) // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr


def save_convergence_model(save_loss, model, epoch):
    print('Saving convergence model at epoch {} with loss {}'.format(epoch, save_loss))
    state = {
        'model' : copy.deepcopy(model),
        'loss'	: save_loss,
        'epoch'	: epoch
    }
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(state, './checkpoint/convergence.t7')


def save_best_acc_model(save_acc, model, epoch):
    print('Saving best acc model at epoch {} with acc in validation set: {}'.format(epoch, save_acc))
    state = {
        'model'	: copy.deepcopy(model),
        'acc'	: save_acc,
        'epoch' : epoch,
    }

    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(state, './checkpoint/best_acc_model.t7')


def train_validate(epoch, optimizer, model, criterion, train_loader, validate_loader):
    optimizer, lr = exp_lr_schedule(args, optimizer, epoch)
    model.train()
    train_loss = 0
    train_acc = 0
    total = 0

    pbar = tqdm(enumerate(train_loader))
    for idx, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        train_acc += predicted.eq(labels).sum().item()

        if idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}\tTrain accuracy: {:.6f}\tLR: {:.3}'.format(
                    epoch + 1, idx * len(images), len(train_loader.dataset),
                    100. * idx / len(train_loader),
                    loss.item(), train_acc / total, lr))

    epoch_acc = train_acc / total
    epoch_loss = train_loss / (idx + 1)

    # Log scalar values (scalar summary)
    info = {'loss': epoch_loss, 'accuracy': epoch_acc}

    for tag, value in info.items():
        train_logger.scalar_summary(tag, value, epoch + 1)

    global save_loss
    # print(save_loss)
    if epoch == 0:
        save_loss = epoch_loss
        save_convergence_model(save_loss, model, epoch)
    else:
        if epoch_loss < save_loss:
            save_loss = epoch_loss
            save_convergence_model(save_loss, model, epoch)

    if (epoch + 1) % args.check_after == 0:
        print('==============================================\n')
        print('==> Validate in epoch {} at LR {:.3}'.format(epoch + 1, lr))
        model.eval()
        validate_loss = 0
        validate_acc = 0
        validate_correct = 0
        total = 0
        pbar = tqdm(enumerate(validate_loader))
        for idx, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            validate_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            validate_correct += predicted.eq(labels).sum().item()

        validate_acc = validate_correct / total
        print('Validation accuracy : {:.6f}'.format(validate_acc))

        # Log scalar values (scalar summary)
        info = {'loss': validate_loss / (idx + 1), 'accuracy': validate_acc}

        for tag, value in info.items():
            val_logger.scalar_summary(tag, value, epoch + 1)

        global save_acc
        # print(save_acc)
        if validate_acc > save_acc:
            save_acc = validate_acc
            save_best_acc_model(save_acc, model, epoch)


dsets = dict()
dsets['train'] = MyTrainDataset(HDF5_TRAIN_PATH, root_dir='./data', train=True, transform=data_transforms['train'])
dsets['val'] = MyTrainDataset(HDF5_TRAIN_PATH, root_dir='./data', train=False, transform=data_transforms['val'])

dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x],
                                   batch_size=args.batch_size,
                                   shuffle=(x != 'val'),
                                   num_workers=1)
    for x in ['train', 'val']
}
##########
print('Load model')
# model = MyDenseNet(depth=args.depth, num_classes=4)

old_model = './checkpoint/convergence.t7'
if args.train_from == 2:
    assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.isfile(old_model), 'Error: no converged model found!'
    print("| Load pretrained at  %s..." % old_model)
    checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
    tmp = checkpoint['model']
    model = unparallelize_model(tmp)
    acc = checkpoint['acc']
    print('Model loaded, previous loss was {:.6f} at epoch {}'.format(checkpoint['loss'], checkpoint['epoch']))
    print('=============================================')
else:
    model = MyResNet(depth=args.depth, num_classes=4)

#############
print('Start training ... ')
model, optimizer = net_frozen(args, model)
model = parallelize_model(model)
# model = params_initializer(model)
criterion = nn.CrossEntropyLoss()
for epoch in range(start_epoch + args.interval):
    train_validate(epoch, optimizer, model, criterion, dset_loaders['train'], dset_loaders['val'])

# if args.inspect:
#
#     checkpoint = torch.load('./checkpoint/convergence.t7')
#     loss = checkpoint['loss']
#     epoch = checkpoint['epoch']
#     print('Model used to predict converges at epoch {} and loss {:.3}'.format(epoch, loss))
#
#     checkpoint = torch.load('./checkpoint/best_acc_model.t7')
#     acc = checkpoint['acc']
#     epoch = checkpoint['epoch']
#     print('Model used to predict has best acc {:3} on validate set at epoch {}'.format(acc, epoch))

