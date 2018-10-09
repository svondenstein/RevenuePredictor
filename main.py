#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import tensorflow as tf

from models.tiramisu import Tiramisu
from utils.preprocess import DataGenerator
from utils.parser import get_config
from utils.utility import create_dirs
from utils.logger import Logger
from utils.rle import prepare_submission
from helpers.trainer import Trainer
from helpers.predicter import Predicter


def main():
    # Get configuration
    config = get_config()

    # Set up test/train environment
    if config.predict or config.train:
        # Set up common environment
        sess = tf.Session()
        data = DataGenerator(config)
        model = Tiramisu(data, config)
        logger = Logger(sess, config)

        if config.train:
            create_dirs([config.model_path])
            trainer = Trainer(sess, model, data, config, logger)
            model.load(sess)
            trainer.train()

        if config.predict:
            predicter = Predicter(sess, model data, config, logger)
            model.load(sess)
            predicter.predict()

    # Prepare submission
    if config.rle:
        prepare_submission(config.predictions, config.submissions, config.image_height, config.image_width)

    print('Done!')


if __name__ == '__main__':
    main()
