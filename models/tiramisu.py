#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import tensorflow as tf

from models.layers import bn_relu_conv, transition_down, transition_up, softmax
from utils.metrics import iou, cross_entropy

class Tiramisu:
    def __init__(self, data_loader, config):
        # Configuration parameters
        self.config = config

        # Step and epoch tensors to use as counters
        self.cur_epoch_tensor = None
        self.increment_cur_epoch_tensor = None
        self.global_step_tensor = None
        self.increment_global_step_tensor = None
        self.global_epoch_tensor = None
        self.increment_global_epoch_tensor = None

        # Initialize step counter
        self.init_global_step()

        # Initialize epoch counter
        self.init_cur_epoch()

        # Initialize global epoch counter
        self.init_global_epoch()

        # Initialize saver
        self.saver = None

        # Initialize data loader
        self.data_loader = data_loader

        # Define local variables
        self.image = None
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
        self.init_saver()

    def build_model(self):
        # Network parameters
        pool = 5
        layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
        filters = 48

        # Inputs to the network
        with tf.variable_scope('inputs'):
            self.image, self.mask = self.data_loader.get_input()
            self.training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.image)
        tf.add_to_collection('inputs', self.mask)
        tf.add_to_collection('inputs', self.training)

        # Network architecture
        with tf.variable_scope('network'):
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
                    l = bn_relu_conv(self.stack, self.config.growth_k, self.config.dropout_percentage, self.training, 'down_dense_block_' + str(i * 10 + j))
                    self.stack = tf.concat([self.stack, l], axis=3, name='down_concat_' + str(i * 10 + j))
                    filters += self.config.growth_k
                skip_connection_list.append(self.stack)
                self.stack = transition_down(self.stack, filters, self.config.dropout_percentage, self.training, 'trans_down_' + str(i))

            skip_connection_list = skip_connection_list[::-1]

            # Bottleneck
            block_to_upsample = []

            # Dense Block
            for j in range(layers_per_block[pool]):
                l = bn_relu_conv(self.stack, self.config.growth_k, self.config.dropout_percentage, self.training, 'bottleneck_dense_' + str(j))
                block_to_upsample.append(l)
                self.stack = tf.concat([self.stack, l], axis=3, name='bottleneck_concat_' + str(j))

            # Upsampling path
            for i in range(pool):
                filters_keep = self.config.growth_k * layers_per_block[pool + i]
                self.stack = transition_up(skip_connection_list[i], block_to_upsample, filters_keep, 'trans_up_' + str(i))

                # Dense block
                block_to_upsample = []
                for j in range(layers_per_block[pool + i + 1]):
                    l = bn_relu_conv(self.stack, self.config.growth_k, self.config.dropout_percentage, self.training, 'up_dense_block_' + str(i + j * 10))
                    block_to_upsample.append(l)
                    self.stack = tf.concat([self.stack, l], axis=3, name='up_concat_' + str(i * 10 + j))

            # Softmax
            with tf.variable_scope('out'):
                self.out = softmax(self.stack, self.config.classes, 'softmax')
                tf.add_to_collection('out', self.out)

        # Operators for the training process
        with tf.variable_scope('loss-acc'):
            self.loss = cross_entropy(self.out, self.mask, self.config.classes)
            self.acc = iou(self.out, self.mask, self.config.classes)

        with tf.variable_scope('train_step'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)

    # Save checkpoint
    def save(self, sess):
        print('Saving model to {}...\n'.format(self.config.model_path))
        self.saver.save(sess, self.config.model_path, self.global_step_tensor)
        print("Model saved.")

    # Load checkpoint
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.model_path)
        if latest_checkpoint:
            print('Loading model checkpoint {} ...\n'.format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded.")

    # Initialize epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = self.cur_epoch_tensor.assign(self.cur_epoch_tensor + 1)

    # Initialize step counter
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = self.global_step_tensor.assign(self.global_step_tensor + 1)

    # Initialize global epoch counter
    def init_global_epoch(self):
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.increment_global_epoch_tensor = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

    # Initialize tensorflow saver used for saving checkpoints
    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)