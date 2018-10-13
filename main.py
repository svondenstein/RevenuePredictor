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
from utils.rle import prepare_submission
from models.trainer import Trainer
from models.predicter import Predicter
from utils.logger import Logger
from tensorboard import default
from tensorboard import program


def main():
    # Set Tensorflow verbosity
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        if config.train:
            print('Creating model save directories...')
            create_dirs([config.model_path, config.log_path])
            print('Initializing TensorBoard...')
            logger = Logger(sess, summary_dir=config.log_path,
                            scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                         'test/loss_per_epoch', 'test/acc_per_epoch'])
            tb = program.TensorBoard(default.PLUGIN_LOADERS, default.get_assets_zip_provider())
            tb.configure(argv=[None, '--logdir', config.log_path])
            tb.main()
            print('Initializing trainer...')
            trainer = Trainer(sess, model, data, config, logger)
            print('Initializing model...')
            model.load(sess)
            print('Training model...')
            trainer.train()
        if config.infer:
            print('Creating prediction save directory...')
            create_dirs([config.prediction_path])
            print('Initializing predicter...')
            predicter = Predicter(sess, model, data, config)
            print('Initializing model...')
            model.load(sess)
            print('Making predictions...')
            predicter.predict()

    # Prepare submission
    if config.rle:
        print('Creating submission save directory...')
        create_dirs([config.submission_path])
        prepare_submission(config.prediction_path, config.submission_path)

    print('Done!')


if __name__ == '__main__':
    main()
