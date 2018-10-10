#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf
import cv2
import os
import numpy as np

def process(image, name, config):
    for i in range(len(image)):
        img = np.reshape(np.argmax(image[i], axis=3), (128, 128, 1))
        cv2.imwrite(os.path.join(config.prediction_path, name[i].decode('utf-8')), 255 * img)
