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
        self.infer_path = self.config.data_path + 'test/images'

        self.image_paths, self.mask_paths, self.name_list = self.get_data_paths_list(self.image_path, self.mask_path)
        self.infer_paths, _, self.infer_name_list = self.get_data_paths_list(self.infer_path, self.infer_path)

        self.num_iterations_test = int((len(self.image_paths) * self.config.validation_split) // self.config.batch_size)
        self.num_iterations_train = int((len(self.image_paths) * (1 - self.config.validation_split))
                                        // self.config.batch_size)
        self.num_iterations_infer = len(self.infer_paths) // self.config.batch_size

        self.images = tf.constant(self.image_paths)
        self.masks = tf.constant(self.mask_paths)
        self.names = tf.constant(self.name_list)
        self.infer_images = tf.constant(self.infer_paths)
        self.infer_names = tf.constant(self.infer_name_list)

        self.dataset = tf.data.Dataset.from_tensor_slices((self.images, self.masks, self.names))
        self.dataset = self.dataset.shuffle(len(self.image_paths))
        self.infer_data = tf.data.Dataset.from_tensor_slices((self.infer_images, self.infer_images, self.infer_names))

        self.test_data = self.dataset.take(tf.cast(len(self.image_paths) * self.config.validation_split, tf.int64))
        self.train_data = self.dataset.skip(tf.cast(len(self.image_paths) * self.config.validation_split, tf.int64))

        self.test_data = process(self.test_data, False, self.config)
        self.train_data = process(self.train_data, True, self.config)
        self.infer_data = process(self.infer_data, False, self.config)

        self.test_data = self.test_data.batch(self.config.batch_size)
        self.train_data = self.train_data.batch(self.config.batch_size)
        self.infer_data = self.infer_data.batch(self.config.batch_size)

        self.iterator = tf.data.Iterator.from_structure(self.test_data.output_types, self.test_data.output_shapes)
        self.training_init_op = self.iterator.make_initializer(self.train_data)
        self.testing_init_op = self.iterator.make_initializer(self.test_data)
        self.infer_init_op = self.iterator.make_initializer(self.infer_data)

    def initialize(self, sess, training):
        if training == 'train':
            sess.run(self.training_init_op)
        elif training == 'test':
            sess.run(self.testing_init_op)
        else:
            sess.run(self.infer_init_op)

    def get_input(self):
        return self.iterator.get_next()

    @staticmethod
    def get_data_paths_list(image_dir, label_dir):
        image_paths = [os.path.join(image_dir, x) for x in os.listdir(
            image_dir) if x.endswith(".png")]
        label_paths = [os.path.join(label_dir, os.path.basename(x))
                      for x in image_paths]
        names = [x.split('/')[-1] for x in image_paths]
        return image_paths, label_paths, names