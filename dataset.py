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

shuffle(train_val_set)
X, Y = zip(*train_val_set)

# Divide the hata into 60% train, 20% validation, 20% test
X_train = X[0:int(0.6 * len(X))]
Y_train = Y[0:int(0.6 * len(Y))]

X_val = X[int(0.6 * len(X)):int(0.8 * len(X))]
Y_val = Y[int(0.6 * len(Y)):int(0.8 * len(Y))]

X_test = X[int(0.8 * len(X)):]
Y_test = Y[int(0.8 * len(Y)):]

img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
data_shape = (0, 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_train_file = tables.open_file(HDF5_TRAIN_PATH, mode='w')
train_storage = hdf5_train_file.create_earray(hdf5_train_file.root, 'X_train', img_dtype, shape=data_shape)
val_storage = hdf5_train_file.create_earray(hdf5_train_file.root, 'X_val', img_dtype, shape=data_shape)
# create the label arrays and copy the labels data in them
hdf5_train_file.create_array(hdf5_train_file.root, 'Y_val', Y_val)

# open a hdf5 file and create earrays
hdf5_test_file = tables.open_file(HDF5_TEST_PATH, mode='w')
test_storage = hdf5_test_file.create_earray(hdf5_test_file.root, 'X_test', img_dtype, shape=data_shape)
# create the label arrays and copy the labels data in them
hdf5_test_file.create_array(hdf5_test_file.root, 'Y_test', Y_test)

X_train_np = np.zeros((len(X_train), 224, 224, 3), dtype=np.uint8)

# loop over train addresses
for i in range(len(X_train)):
    # print how many images are saved every 1000 images
    if i % 50 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(X_train)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = X_train[i]
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.equalizeHist(img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    # img = img.reshape(img.shape + (1,))
    # img = img.transpose(2, 0, 1)
    # save the image
    X_train_np[i, :, :, :] = img

if args.aug_multipliers > 0:
    aug_X_train = np.vstack([rand_similarity_trans(img, args.aug_multipliers) for img in X_train_np])
    aug_Y_train = np.repeat(Y_train, args.aug_multipliers)
    X_train_np = np.append(X_train_np, aug_X_train, axis=0)
    Y_train = np.append(Y_train, aug_Y_train, axis=0)
    print('Number of training samples after augmentations: {}'.format(len(X_train_np)))

hdf5_train_file.create_array(hdf5_train_file.root, 'Y_train', Y_train)
train_storage.append(X_train_np)

# loop over val addresses
for i in range(len(X_val)):
    # print how many images are saved every 1000 images
    if i % 50 == 0 and i > 1:
        print('Val data: {}/{}'.format(i, len(X_val)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = X_val[i]
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.equalizeHist(img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    # img = img.reshape(img.shape + (1,))
    # img = img.transpose(2, 0, 1)
    # save the image
    val_storage.append(img[None])

# loop over test addresses
for i in range(len(X_test)):
    # print how many images are saved every 1000 images
    if i % 50 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(X_test)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = X_test[i]
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.equalizeHist(img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    # img = img.reshape(img.shape + (1,))
    # img = img.transpose(2, 0, 1)
    # save the image
    test_storage.append(img[None])

hdf5_train_file.close()
hdf5_test_file.close()
