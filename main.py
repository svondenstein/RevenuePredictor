#!/usr/bin/env python3
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
        if config.train:
            print('Creating model save directories...')
            create_dirs([config.model_path])
            print('Initializing trainer...')
            trainer = Trainer(sess, model, data, config)
            print('Initializing model...')
            model.load(sess)
            print('Training model...')
            trainer.train()
        if config.infer:
            print('Initializing predicter...')
            predicter = Predicter(sess, model, data, config)
            print('Initializing model...')
            model.load(sess)
            print('Making predictions...')
            predicter.predict()
        if config.optimize:
            print('Initializing optimizer...')
            optimizer = HyperEngineOptimizer(model, data, config)
            print('Initializing model...')
            model.load(sess)
            print('Optimizing...')
            optimizer.optimize()

    # Prepare submission
    if config.rle:
        prepare_submission(config.prediction_path, config.submission_path, 101, 101)

    print('Done!')


if __name__ == '__main__':
    main()
