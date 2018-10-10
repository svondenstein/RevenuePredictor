#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import tensorflow as tf

@staticmethod
# Apply batch normalization, ReLu, Convolution, and Dropout on the inputs
def bn_relu_conv(inputs, filters, dropout, training, name, filter_size=3):
    with tf.variable_scope(name):
        l = tf.layers.batch_normalization(inputs, training=training, name=name + '_bn')
        l = tf.layers.conv2d(l,
                             filters=filters,
                             kernel_size=filter_size,
                             padding='SAME',
                             dilation_rate=filter_size,
                             activation='relu',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             name=name+'_conv' + str(filter_size) + 'x' + str(filter_size))
        if dropout != 0.0:
            l = tf.layers.dropout(l, rate=dropout, training=training, name=name+'_dropout')

    return l


@staticmethod
# Apply bn_relu_conv layer and then a max pooling
def transition_down(inputs, filters, dropout, training, name):
    with tf.variable_scope(name):
        l = bn_relu_conv(inputs, filters, dropout, training, name + 'transition_down', filter_size=1)
        l = tf.layers.max_pooling2d(l, 2, 2, name=name + 'max_pool')

    return l

@staticmethod
# Perform upsampling on block by factor 2 and concatenates it with skip connection
def transition_up(skip_connection, block, filters, training, name):
    with tf.variable_scope(name):
        l = tf.concat(block, name=name + 'concat_up_1')
        l = tf.layers.conv2d_transpose(l,
                                       filters=filters,
                                       kernel_size=3,
                                       strides=2,
                                       activation='relu',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       name=name+'_trans_conv_3x3')
        l = tf.concat([l, skip_connection], name=name + 'concat_up_2')

    return l


@staticmethod
# Perform 1x1 convolution followed by softmax nonlinearity
def softmax(stack, classes, training, name):
    with tf.variable_scope(name):
        l = tf.layers.conv2d(stack,
                             filters=filters,
                             kernel_size=1,
                             padding='SAME',
                             activation='relu',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             name='last_conv1x1')
    # Softmax?
    return l