#
# Stephen Vondenstein, Matthew Buckley
# 10/12/2018
#
from helpers.data_generator import DataGenerator
import cv2
import os
import tensorflow as tf
from utils.parser import get_args
from helpers.postprocess import data_crop
from helpers.preprocess import tile_data

def main():
    config = get_args()
    data_loader = DataGenerator(config)
    with tf.Session() as sess:
        data_loader.initialize(sess, 'train')
        image, mask, name = sess.run(data_crop(*tile_data(*data_loader.get_input())))
        print(image.shape)
        for i in range(16):
            print(name[i].decode())
            cv2.imwrite(os.path.join('./image_tests/images', name[i].decode('utf-8')), 255 * image[i, :, :])
            cv2.imwrite(os.path.join('./image_tests/masks', name[i].decode('utf-8')), 255 * mask[i, :, :])
            print(name[i].decode())


if __name__ == '__main__':
    main()