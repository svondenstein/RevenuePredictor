#
# Stephen Vondenstein, Matthew Buckley
# 10/12/2018
#
from helpers.data_generator import DataGenerator
import cv2
import os, subprocess
import tensorflow as tf

from utils.parser import get_args
from utils.utility import create_dirs
from utils.rle import prepare_submission
from helpers.postprocess import data_crop
from helpers.preprocess import tile_data

def main():
    config = get_args()
    create_dirs(['./image_tests/',
                 './image_tests/tiled/', './image_tests/tiled/images/', './images_tests/tiled/masks/',
                 './image_tests/detiled', './image_tests/detiled/images/', './images_tests/detiled/masks/',
                 config.submission_path])
    data_loader = DataGenerator(config)
    with tf.Session() as sess:
        data_loader.initialize(sess, 'train')
        for i in range(data_loader.num_iterations_train):
            image, mask, name = sess.run(tile_data(*data_loader.get_input()))
            for i in range(config.batch_size):
                cv2.imwrite(os.path.join('./image_tests/tiled/images', name[i].decode('utf-8')), 255 * image[i, :, :])
                cv2.imwrite(os.path.join('./image_tests/tiled/masks', name[i].decode('utf-8')), 255 * mask[i, :, :])
            image = tf.convert_to_tensor(image)
            mask = tf.convert_to_tensor(mask)
            name = tf.convert_to_tensor(name)
            image, mask, name = sess.run(data_crop(image, mask, name))
            for i in range(config.batch_size):
                cv2.imwrite(os.path.join('./image_tests/detiled/images', name[i].decode('utf-8')), 255 * image[i, :, :])
                cv2.imwrite(os.path.join('./image_tests/detiled/masks', name[i].decode('utf-8')), 255 * mask[i, :, :])

        rle = prepare_submission('./image_tests/detiled/', config.submission_path, 'iotest')
        subprocess.Popen("sort ./data/train.csv > ./image_tests/source.csv", shell=True)
        subprocess.Popen("sort " + rle + " > ./image_tests/test.csv", shell=True)
        print(subprocess.Popen("diff -as ./data/test.csv ./data/test2.csv", shell=True,
                               stdout=subprocess.PIPE).stdout.readline().decode(), end='')

        print('Done!')

if __name__ == '__main__':
    main()