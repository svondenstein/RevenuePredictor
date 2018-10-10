#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import os
import tensorflow as tf

from models.tiramisu import Tiramisu
from helpers.data_generator import DataGenerator
from utils.parser import get_args
from utils.utility import create_dirs
from utils.logger import Logger
from utils.rle import prepare_submission
from models.trainer import Trainer


def main():
    # Set Tensorflow verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Get configuration
    config = get_args()

    # Set up test/train environment
    if config.infer or config.train:
        # Set up common environment
        sess = tf.Session()
        print('Loading data...')
        data = DataGenerator(config)
        print('Building model...')
        model = Tiramisu(data, config)
        print('Initializing Tensorboard...')
        logger = Logger(sess, summary_dir=config.summary_path,
                        scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                     'test/loss_per_epoch', 'test/acc_per_epoch'])

        print('Creating model save directories...')
        create_dirs([config.model_path, config.summary_path])
        print('Initializing trainer...')
        trainer = Trainer(sess, model, data, config, logger)
        print('Initializing model...')
        model.load(sess)
        print('Training model...')
        trainer.train()

    # Prepare submission
    if config.rle:
        prepare_submission(config.predictions, config.submissions, 101, 101)

    print('Done!')


if __name__ == '__main__':
    main()
