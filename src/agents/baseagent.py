#
# Stephen Vondenstein, Matthew Buckley
# 10/11/2018
#

# from tqdm import tqdm
import tensorflow as tf
from src.helpers.data_generator import DataGenerator


class BaseAgent:
    def __init__(self, config):
        # Set up config and data
        self.config = config
        print('Loading data...')
        self.data_loader = DataGenerator(config)

        # Initialize saver
        self.saver = None
        self.init_saver()

    # Save checkpoint
    def save(self, sess):
        print('Saving model to {}...'.format(self.config.model_path))
        self.saver.save(sess, self.config.model_path + 'epoch', self.global_epoch_tensor)
        print("Model saved.")

    # Load checkpoint
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.model_path)
        if latest_checkpoint:
            print('Loading model checkpoint {} ...'.format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded.")

    # Initialize tensorflow saver used for saving checkpoints
    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
