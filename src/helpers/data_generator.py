#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import os
import pandas as pan

import tensorflow as tf
from src.helpers.preprocess import process

# Some of this needs to change when validation split is implemented
class DataGenerator:
    def __init__(self, config):
        self.config = config

        # Get paths for data directories
        self.image_path = os.path.join(self.config.data_path,'train/images/')
        self.mask_path = os.path.join(self.config.data_path,'train/masks')
        self.infer_path = os.path.join(self.config.data_path,'test/images')
        self.depth_file = os.path.join(self.config.data_path,'depths.csv')

        # Get list of images in each data directory
        self.image_paths, self.mask_paths, self.name_list, self.depths_list = self.get_data_paths_list(self.image_path, self.mask_path, self.depth_file)
        self.infer_paths, _, self.infer_name_list, self.infer_depths_list = self.get_data_paths_list(self.infer_path, self.infer_path, self.depth_file)

        # Calculate size of each data subset
        self.test_size = int((len(self.image_paths)) * self.config.validation_split)
        self.train_size = int((len(self.image_paths)) * (1 - self.config.validation_split))
        self.infer_size = len(self.infer_paths)

        # Calculate the max number of iterations based on batch size
        self.num_iterations_test = self.test_size // self.config.batch_size if \
            self.config.validation_split != 0.0 else self.train_size // self.config.batch_size
        self.num_iterations_train = (self.train_size // self.config.batch_size) * 7 if self.config.augment else self.train_size // self.config.batch_size
        self.num_iterations_infer = self.infer_size // self.config.batch_size

        # Create tensors containing image paths
        self.images = tf.constant(self.image_paths)
        self.masks = tf.constant(self.mask_paths)
        self.names = tf.constant(self.name_list)
        self.depths = tf.constant(self.depths_list)

        self.infer_images = tf.constant(self.infer_paths)
        self.infer_names = tf.constant(self.infer_name_list)
        self.infer_depths = tf.constant(self.infer_depths_list)

        # Create train and infer datasets from the individual tensors
        self.dataset = tf.data.Dataset.from_tensor_slices((self.images, self.masks, self.names, self.depths))
        self.dataset = self.dataset.shuffle(len(self.image_paths))
        self.infer_data = tf.data.Dataset.from_tensor_slices((self.infer_images, self.infer_images, self.infer_names, self.infer_depths))

        # Split training set for validation
        self.test_data = self.dataset.take(tf.cast(len(self.image_paths) * self.config.validation_split, tf.int64))
        self.train_data = self.dataset.skip(tf.cast(len(self.image_paths) * self.config.validation_split, tf.int64))

        # Preprocess datasets
        self.test_data = process(self.test_data, False, self.config, self.test_size) if \
            self.config.validation_split != 0.0 else process(self.train_data, False, self.config, self.train_size)
        self.train_data = process(self.train_data, True, self.config, self.train_size)
        self.infer_data = process(self.infer_data, False, self.config, self.infer_size)

        # Create iterator for train and infer datasets
        self.data_iterator = tf.data.Iterator.from_structure(self.train_data.output_types,
                                                              self.train_data.output_shapes)
        self.val_iterator = tf.data.Iterator.from_structure(self.test_data.output_types,
                                                             self.test_data.output_shapes)
        self.training_init_op = self.data_iterator.make_initializer(self.train_data)
        self.testing_init_op = self.val_iterator.make_initializer(self.test_data)
        self.infer_init_op = self.data_iterator.make_initializer(self.infer_data)

    # Initialize the iterater based on context
    def initialize(self, sess, training):
        if training == 'train':
            sess.run(self.training_init_op)
        elif training == 'test':
            sess.run(self.testing_init_op)
        else:
            sess.run(self.infer_init_op)

    # Get input methods for input and validation data
    def get_data(self):
        return self.data_iterator.get_next()

    def get_val(self):
        return self.val_iterator.get_next()

    # Return list of images, masks, and names for a given data directory
    @staticmethod
    def get_data_paths_list(image_dir, label_dir, depth_file):
        image_paths = [os.path.join(image_dir, x) for x in os.listdir(
            image_dir) if x.endswith(".png")]
        label_paths = [os.path.join(label_dir, os.path.basename(x))
                      for x in image_paths]
        names = [x.split('/')[-1] for x in image_paths]
        csv = pan.read_csv(depth_file, index_col='id')
        depths = [csv.loc[name.split('.')[0],'z'] for name in names]
        return image_paths, label_paths, names, depths
