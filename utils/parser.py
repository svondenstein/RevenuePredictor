#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--augment',
                        action='store_true',
                        default='store_false',
                        help='Augment data to increase the size and variation of the training set')

    parser.add_argument('-b', '--batch_size',
                        default=16,
                        help='Batch size for use in training',
                        type=int)

    parser.add_argument('-c', '--classes',
                        default=2,
                        help='Number of classes',
                        type=int)

    parser.add_argument('-d', '--data_path',
                        default='./data/',
                        help='Path to data directory')

    parser.add_argument('-dp', '--dropout_percentage',
                        default=0.2,
                        help='Dropout percentage for dropout layer')

    parser.add_argument('-e', '--epochs',
                        default=32,
                        help='Number of epochs for training',
                        type=int)

    parser.add_argument('-i', '--infer',
                        action='store_true',
                        default='store_false',
                        help='Perform inference')

    parser.add_argument('-k', '--growth_k',
                        default=16,
                        help='Growth rate',
                        type=int)

    parser.add_argument('-lr', '--learning_rate',
                        default=1e-3,
                        help='Learning rate for optimizer',
                        type=float)

    parser.add_argument('-m', '--model_path',
                        default='./models/tiramisu/',
                        help='Directory to save trained models')

    parser.add_argument('-mk', '--max_to_keep',
                        default=0,
                        help='Maximum number of models to save')

    parser.add_argument('-p', '--prediction_path',
                        default='./predictions/',
                        help='Directory to save predicted masks')

    parser.add_argument('-r', '--rle',
                        action='store_true',
                        default='store_false',
                        help='Compute RLE and prepare submission')

    parser.add_argument('-s', '--submission_path',
                        default='./submissions/',
                        help='Directory to save submissions')

    parser.add_argument('-sp', '--summary_path',
                        default='./models/summaries/',
                        help='Directory to save Tensorboard summaries')

    parser.add_argument('-t', '--train',
                        action='store_true',
                        default='store_false',
                        help='Perform model training')

    parser.add_argument('-v', '--validation_split',
                        default=0.25,
                        help='Validation split for training',
                        type=float)

    FLAGS = parser.parse_args()
    return FLAGS
