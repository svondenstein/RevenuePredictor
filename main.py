#!/usr/bin/env python3
#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import os

from src.models.tiramisu import Tiramisu
from src.utils.parser import get_args
from src.utils.utility import create_dirs, generate_params
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
        print('Creating save directories...')
        create_dirs([config.optimizer_path,config.model_path])
        print('Initializing optimizer...')
        # optimizer = HyperEngineOptimizer(config)
        print('Optimizing...')
        # optimizer.optimize()

    # Set up test/train environment
    if config.infer or config.train:
        print('Building model...')
        params = generate_params(config)
        model = Tiramisu(params)
        if config.train:
            print('Creating save directories...')
            create_dirs([config.model_path])
            print('Initializing trainer...')
            trainer = Trainer(config)
            print('Training model...')
            trainer.train()
        if config.infer:
            print('Creating save directories...')
            create_dirs([config.prediction_path])
            print('Initializing predicter...')
            predicter = Predicter(config)
            predicter.predict()

    # Prepare submission
    if config.rle:
        print('Creating submission save directory...')
        create_dirs([config.submission_path])
        prepare_submission(config.prediction_path, config.submission_path, 'submission')

    print('Done!')


if __name__ == '__main__':
    main()
