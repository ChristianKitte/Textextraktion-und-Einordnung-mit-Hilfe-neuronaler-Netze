"""
This file contains source code from another GitHub project. The comments made there apply. The used
source code is licensed under the MIT License. The license text and a detailed reference can be
found in the license subfolder. Many thanks to the author of the code.

For reasons of clarity unneeded parts of the original code were not taken over. The original project
can be found on the https://github.com/kurapan/CRNN page.

The remaining code has been left in its original state.
"""

import cv2
import numpy as np


def pad_image(img, img_size, nb_channels):
    # img_size : (width, height)
    # loaded_img_shape : (height, width)
    img_reshape = cv2.resize(img, (int(img_size[1] / img.shape[0] * img.shape[1]), img_size[1]))
    if nb_channels == 1:
        padding = np.zeros((img_size[1], img_size[0] - int(img_size[1] / img.shape[0] * img.shape[1])), dtype=np.int32)
    else:
        padding = np.zeros((img_size[1], img_size[0] - int(img_size[1] / img.shape[0] * img.shape[1]), nb_channels),
                           dtype=np.int32)
    img = np.concatenate([img_reshape, padding], axis=1)
    return img


def resize_image(img, img_size):
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    img = np.asarray(img)
    return img
