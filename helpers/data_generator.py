#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import os

import tensorflow as tf

# Some of this needs to change when validation split is implemented
class DataGenerator:
    def __init__(self, config):
        self.config = config

        self.image_path = self.config.data_path + 'train/images/'
        self.mask_path = self.config.data_path + 'train/masks'

        self.image_paths, self.mask_paths = self.get_data_paths_list(self.image_path, self.mask_path)

        self.images = tf.constant(self.image_paths)
        self.masks = tf.constant(self.mask_paths)

        self.num_iterations_train = len(self.image_paths) // self.config.batch_size
        self.num_iterations_test = self.num_iterations_train

        self.dataset = tf.data.Dataset.from_tensor_slices((self.images, self.masks))
        self.dataset = self.dataset.map(DataGenerator.parse_data, num_parallel_calls=self.config.batch_size)
        self.dataset = self.dataset.map(DataGenerator.resize_data, num_parallel_calls=self.config.batch_size)
        self.dataset = self.dataset.map(DataGenerator.normalize_data, num_parallel_calls=self.config.batch_size)
        self.dataset = self.dataset.shuffle(1000, reshuffle_each_iteration=False)
        self.dataset = self.dataset.batch(self.config.batch_size)
        # self.dataset = self.dataset.repeat(1)

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        self.training_init_op = self.iterator.make_initializer(self.dataset)

    @staticmethod
    def parse_data(image_paths, label_paths):
        image_content = tf.read_file(image_paths)
        images = tf.image.decode_png(image_content, channels=1)

        mask_content = tf.read_file(label_paths)
        masks = tf.image.decode_png(mask_content, channels=1)

        return images, masks

    @staticmethod
    def resize_data(image, mask):
        image = tf.image.resize_images(image, [128, 128])
        mask = tf.image.resize_images(mask, [128, 128])

        return image, mask

    @staticmethod
    def normalize_data(image, mask):
        image = image / 255.0
        mask = mask / 255.0

        return image, mask

    def initialize(self, sess):
        sess.run(self.training_init_op)

    def get_input(self):
        return self.iterator.get_next()

    @staticmethod
    def get_data_paths_list(image_dir, label_dir):
        image_paths = [os.path.join(image_dir, x) for x in os.listdir(
            image_dir) if x.endswith(".png")]
        label_paths = [os.path.join(label_dir, os.path.basename(x))
                      for x in image_paths]

        return image_paths, label_paths