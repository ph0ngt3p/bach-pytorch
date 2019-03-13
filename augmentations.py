import random
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from constants import *

ia.seed(1)


def augment(images):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent = (0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    images_aug = seq.augment_images(images)
    return images_aug


def rand_similarity_trans(image, n):
    """
    apply random similarity transformation to the image, and return
    n transformed images
    """
    output_images = np.uint8(np.zeros((n, INPUT_SIZE, INPUT_SIZE, 3)))

    for i in range(n):
        angle = random.uniform(-15, 15)  # rotation

        s = random.uniform(0.9, 1.1)  # scale

        rows, cols = image.shape[0:2]
        image_center = (rows / 2.0, cols / 2.0)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
        M_rot = np.vstack([rot_mat, [0, 0, 1]])

        tx = random.uniform(-2, 2)  # translation along x axis
        ty = random.uniform(-2, 2)  # translation along y axis
        M_tran = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

        M = np.matrix(M_tran) * np.matrix(M_rot)

        M = np.float32(M[:2][:])  # similarity transform

        tmp = cv2.warpAffine(image, M, (cols, rows))
        # tmp = tmp.reshape(tmp.shape + (1, ))
        output_images[i, :, :, :] = tmp
        # print(output_images[i, :, :, :].shape)

        # cv2.equalizeHist(image, image)

    return output_images

# def test():
#     img = cv2.imread('/home/tep/PycharmProjects/bach-pytorch/data/Benign/b001.tif')
#     img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# test()
