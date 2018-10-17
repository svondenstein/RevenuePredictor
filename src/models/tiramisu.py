#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import tensorflow as tf


from src.models.layers import bn_relu_conv, transition_down, transition_up, softmax
from src.utils.metrics import iou, cross_entropy
from src.helpers.preprocess import tile_image
from src.helpers.postprocess import tile_crop


class Tiramisu:
    def __init__(self, config):
        # Configuration parameters
        self.config = config

        # Define local variables
        self.image = None
        self.image_name = None
        self.mask = None
        self.training = None
        self.out = None
        self.stack = None
        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None

        # Build model and initialize saver
        self.build_model()

    def build_model(self):
        # Network parameters
        pool = 5
        layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
        filters = 48

        # Inputs to the network
        with tf.variable_scope('inputs'):
            # self.image, self.mask, self.image_name = self.data_loader.get_input()
            self.training = tf.placeholder(tf.bool, name='Training_flag')
            self.image = tf.placeholder(tf.float32, shape=[None, 101, 101, 2],name='input')
            self.mask = tf.placeholder(tf.int32, shape=[None, 101, 101, 1], name='label') #Should this be boolean?
        tf.add_to_collection('inputs', self.image)
        tf.add_to_collection('inputs', self.mask)
        tf.add_to_collection('inputs', self.training)
        # tf.add_to_collection('inputs', self.image_name)

        # Network architecture
        with tf.variable_scope('network'):
            self.image = tile_image(self.image)
            # First convolution
            self.stack = tf.layers.conv2d(self.image,
                                          filters=filters,
                                          kernel_size=3,
                                          padding='SAME',
                                          activation='relu',
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                          name='first_conv3x3')

            # Downsampling path
            skip_connection_list = []

            for i in range(pool):
                # Dense block
                for j in range(layers_per_block[i]):
                    l = bn_relu_conv(self.stack, self.config['growth_k'], self.config['conv']['dropout'], self.training, 'down_dense_block_' + str(i * 10 + j))
                    self.stack = tf.concat([self.stack, l], axis=3, name='down_concat_' + str(i * 10 + j))
                    filters += self.config['growth_k']
                skip_connection_list.append(self.stack)
                self.stack = transition_down(self.stack, filters, self.config['conv']['dropout'], self.training, 'trans_down_' + str(i))

            skip_connection_list = skip_connection_list[::-1]

            # Bottleneck
            block_to_upsample = []

            # Dense Block
            for j in range(layers_per_block[pool]):
                l = bn_relu_conv(self.stack, self.config['growth_k'], self.config['conv']['dropout'], self.training, 'bottleneck_dense_' + str(j))
                block_to_upsample.append(l)
                self.stack = tf.concat([self.stack, l], axis=3, name='bottleneck_concat_' + str(j))

            # Upsampling path
            for i in range(pool):
                filters_keep = self.config['growth_k'] * layers_per_block[pool + i]
                self.stack = transition_up(skip_connection_list[i], block_to_upsample, filters_keep, 'trans_up_' + str(i))

                # Dense block
                block_to_upsample = []
                for j in range(layers_per_block[pool + i + 1]):
                    l = bn_relu_conv(self.stack, self.config['growth_k'], self.config['conv']['dropout'], self.training, 'up_dense_block_' + str(i + j * 10))
                    block_to_upsample.append(l)
                    self.stack = tf.concat([self.stack, l], axis=3, name='up_concat_' + str(i * 10 + j))

            # Softmax
            with tf.variable_scope('out'):
                self.out = softmax(self.stack, self.config['classes'], 'softmax')
                self.out = tile_crop(self.out)
                tf.add_to_collection('out', self.out)

        # Operators for the training process
        with tf.variable_scope('loss-acc'):
            self.loss = cross_entropy(self.out, self.mask, self.config['classes'], name='loss')
            self.acc = iou(self.out, self.mask, name='accuracy')

        with tf.variable_scope('train_step'):
            self.optimizer = tf.train.AdamOptimizer(self.config['optimizer']['learning_rate'])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, name='minimize')

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)
