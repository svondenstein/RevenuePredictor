#!/usr/bin/env python3
#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import os

from src.models.tiramisu import Tiramisu
from src.helpers.data_generator import DataGenerator
from src.utils.parser import get_args
from src.utils.utility import create_dirs
from src.utils.rle import prepare_submission
from src.agents.trainer import Trainer
from src.agents.predicter import Predicter
from src.agents.optimizers.tiramisu_hyper import HyperEngineOptimizer


def main():
    # Set Tensorflow verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Get configuration
    config = get_args()

    # Optimize the parameters
    if config.optimize:
        print('Initializing optimizer...')
        optimizer = HyperEngineOptimizer(config)
        print('Optimizing...')
        optimizer.optimize()

    # Set up test/train environment
    if config.infer or config.train:
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
            trainer.load(sess)
            print('Training model...')
            trainer.train()
        if config.infer:
            print('Initializing predicter...')
            predicter = Predicter(sess, model, data, config)
            print('Initializing model...')
            predicter.load(sess)
            print('Making predictions...')
            predicter.predict()

    # Prepare submission
    if config.rle:
        prepare_submission(config.prediction_path, config.submission_path, 101, 101)

    print('Done!')


if __name__ == '__main__':
    main()
