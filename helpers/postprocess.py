#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import cv2
import os
import numpy as np
import tensorflow as tf


def process(image, name, config):
    for i in range(image[0].shape[0]):
        img = np.reshape(np.argmax(image[0], axis=3), (image[0].shape[0],image[0].shape[1],image[0].shape[2]))
        cv2.imwrite(os.path.join(config.prediction_path, name[i].decode('utf-8')), 255 * img[i, :, :])


# de-tile the image
def tile_crop(image):
    out = tf.image.crop_to_bounding_box(image, 13, 13, 101, 101)
    return out


# mostly for testing
def data_crop(image, mask, name):
    image = tile_crop(image)
    mask = tile_crop(mask)
    return image, mask, name
