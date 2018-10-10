#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import os

import tensorflow as tf
from helpers.preprocess import process

# Some of this needs to change when validation split is implemented
class DataGenerator:
    def __init__(self, config):
        self.config = config

        self.image_path = self.config.data_path + 'train/images/'
        self.mask_path = self.config.data_path + 'train/masks'

        self.image_paths, self.mask_paths = self.get_data_paths_list(self.image_path, self.mask_path)

        self.num_iterations_test = (len(self.image_paths) * self.config.validation_split) // self.config.batch_size
        self.num_iterations_train = (len(self.image_paths) * (1 - self.config.validation_split)) // self.config.batch_size

        self.images = tf.constant(self.image_paths)
        self.masks = tf.constant(self.mask_paths)

        self.dataset = tf.data.Dataset.from_tensor_slices((self.images, self.masks))
        self.dataset = self.dataset.shuffle(len(self.image_paths))

        self.test_data = self.dataset.take(tf.cast(len(self.image_paths) * self.config.validation_split, tf.int64))
        self.train_data = self.dataset.skip(tf.cast(len(self.image_paths) * self.config.validation_split, tf.int64))

        self.test_data = process(self.test_data, False, self.config)
        self.train_data = process(self.train_data, True, self.config)

        self.test_data = self.test_data.batch(self.config.batch_size)
        self.train_data = self.train_data.batch(self.config.batch_size)

        self.iterator = tf.data.Iterator.from_structure(self.test_data.output_types, self.test_data.output_shapes)
        self.training_init_op = self.iterator.make_initializer(self.train_data)
        self.testing_init_op = self.iterator.make_initializer(self.test_data)

    def initialize(self, sess, training):
        if training:
            sess.run(self.training_init_op)
        else:
            sess.run(self.testing_init_op)

    def get_input(self):
        return self.iterator.get_next()

    @staticmethod
    def get_data_paths_list(image_dir, label_dir):
        image_paths = [os.path.join(image_dir, x) for x in os.listdir(
            image_dir) if x.endswith(".png")]
        label_paths = [os.path.join(label_dir, os.path.basename(x))
                      for x in image_paths]

        return image_paths, label_paths