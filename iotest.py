#
# Stephen Vondenstein, Matthew Buckley
# 10/12/2018
#
from helpers.data_generator import DataGenerator
import cv2
import os, subprocess
import tensorflow as tf

from tqdm import tqdm
from utils.parser import get_args
from utils.utility import create_dirs
from utils.rle import prepare_submission
from helpers.postprocess import data_crop
from helpers.preprocess import tile_data

def main():
    # Set Tensorflow verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

    config = get_args()
    print('Creating save directories...')
    create_dirs(['./image_tests/', './image_tests'
                 './image_tests/tiled/', './image_tests/tiled/images/', './image_tests/tiled/masks/',
                 './image_tests/detiled', './image_tests/detiled/images/', './image_tests/detiled/masks/',
                 config.submission_path])
    print('Initializing data loader...')
    data_loader = DataGenerator(config)

    # Progress bar
    batches = tqdm(range(data_loader.num_iterations_train), total=data_loader.num_iterations_train,
              desc="Processing Batches ")

    # Use session to evaluate real tensors
    with tf.Session() as sess:
        # Initialize data generator
        data_loader.initialize(sess, 'debug')
        for i in batches:
            # Get a batch and tile it
            image, mask, name = sess.run(tile_data(*data_loader.get_input()))
            # Save tiled images
            for i in range(config.batch_size):
                cv2.imwrite(os.path.join('./image_tests/tiled/images', name[i].decode('utf-8')), 255 * image[i, :, :])
                cv2.imwrite(os.path.join('./image_tests/tiled/masks', name[i].decode('utf-8')), 255 * mask[i, :, :])
            # Convert images back to tensors because that's how the de-tiler expects them
            image = tf.convert_to_tensor(image)
            mask = tf.convert_to_tensor(mask)
            name = tf.convert_to_tensor(name)
            # De-tile images
            image, mask, name = sess.run(data_crop(image, mask, name))
            # Save de-tiled images
            for i in range(config.batch_size):
                cv2.imwrite(os.path.join('./image_tests/detiled/images', name[i].decode('utf-8')), 255 * image[i, :, :])
                cv2.imwrite(os.path.join('./image_tests/detiled/masks', name[i].decode('utf-8')), 255 * mask[i, :, :])

    # Close progress bar
    batches.close()
    # Compute RLEs for saved masks
    rle = prepare_submission('./image_tests/detiled/masks/', config.submission_path, 'iotest')
    print('Comparing RLE values...')
    subprocess.Popen("sort ./data/train.csv > ./image_tests/source.csv", shell=True)
    subprocess.Popen("sort " + rle + " > ./image_tests/test.csv", shell=True)
    # Compare RLEs for saved masks
    print(subprocess.Popen("diff -as ./image_tests/source.csv ./image_tests/test.csv", shell=True,
                           stdout=subprocess.PIPE).stdout.readline().decode(), end='')

    print('Done!')


if __name__ == '__main__':
    main()