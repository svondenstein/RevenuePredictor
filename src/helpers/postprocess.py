#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import cv2
import os
import numpy as np
import tensorflow as tf
import math


def process(image, name, config):
    #print(name.shape)
    for i in range(len(image[0])):
        img = np.argmax(image[0][i], axis=2)
        print(name[i])
        cv2.imwrite(os.path.join(config.prediction_path, name[i].decode('utf-8')), 255 * img)


# de-tile the image
def tile_crop(image):
    offset = math.floor((128-101)/2)  # Half the margin between the sizes
    out = tf.image.crop_to_bounding_box(image, offset, offset, 101, 101)
    return out


# mostly for testing
def data_crop(image, mask, name):
    image = tile_crop(image)
    mask = tile_crop(mask)
    return image, mask, name
