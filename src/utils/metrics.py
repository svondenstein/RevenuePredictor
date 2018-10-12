#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf


# TODO: Fix this, I think it's incorrect.
def iou(prediction, mask, num_classes, name='accuracy'):
    inter = tf.reduce_sum(tf.multiply(prediction, mask))
    union = tf.reduce_sum(tf.subtract(tf.add(prediction, mask), tf.multiply(prediction, mask)))
    mean_iou = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union), name=name)

    return mean_iou


def cross_entropy(prediction, mask, num_classes, name='loss'):
    mask = tf.cast(mask, tf.int32)
    prediction = tf.reshape(prediction, (-1, num_classes))
    mask = tf.reshape(mask, [-1])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=mask)
    loss = tf.reduce_mean(ce, name=name)

    return loss
