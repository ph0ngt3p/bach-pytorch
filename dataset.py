import cv2
import os
import sys
import glob
from random import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import torch
import tables
from augmentations import rand_similarity_trans
import argparse
from constants import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--aug_multipliers', '-a', default=0, type=int, help='Augmentations multiplier')
args = parser.parse_args()

labels = {
    'Normal': 0,
    'Benign': 1,
    'InSitu': 2,
    'Invasive': 3
}

train_val_set = []

for label in [name for name in os.listdir(TRAINVAL_DATA_PATH) if os.path.isdir(os.path.join(TRAINVAL_DATA_PATH, name))]:
    paths = glob.glob(os.path.join(TRAINVAL_DATA_PATH, label, '*.tif'))
    for path in paths:
        subset = (path, labels[label])
        train_val_set.append(subset)

# shuffle(train_val_set)
X, y = zip(*train_val_set)

X_np = np.zeros((len(X), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

# loop over addresses
for i in range(len(X)):
    # print how many images are saved every 1000 images
    if i % 50 == 0 and i > 1:
        print('Processing: {}/{}'.format(i, len(X)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = X[i]
    img = cv2.imread(addr)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    # save the image
    X_np[i, :, :, :] = img

# Divide the data into 300 train, 100 test
X_train, X_test, y_train, y_test = train_test_split(X_np, y, test_size=0.25, random_state=42, stratify=y)

if args.aug_multipliers > 0:
    aug_X_train = np.vstack([rand_similarity_trans(img, args.aug_multipliers) for img in X_train])
    aug_y_train = np.repeat(y_train, args.aug_multipliers)
    X_train = np.append(X_train, aug_X_train, axis=0)
    y_train = np.append(y_train, aug_y_train, axis=0)
    print('Number of training samples after augmentations: {}'.format(len(X_train)))

np.savez('./data/train_val.npz', X_train=X_train, y_train=y_train)
np.savez('./data/test.npz', X_test=X_test, y_test=y_test)

# import collections
# print(collections.Counter(y_test))
# print(collections.Counter(y_train))
