#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf


# TODO: Fix this, I think it's incorrect.
def iou(prediction, mask, num_classes):
    prediction = tf.cast(tf.expand_dims(tf.argmax(prediction, axis=3), [-1]), tf.float32)
    inter = tf.reduce_sum(tf.multiply(prediction, mask))
    union = tf.reduce_sum(tf.add(prediction, mask))
    mean_iou = tf.reduce_mean(tf.divide(inter, tf.subtract(union, inter)))

    return mean_iou


def cross_entropy(prediction, mask, num_classes):
    mask = tf.cast(mask, tf.int32)
    prediction = tf.reshape(prediction, (-1, num_classes))
    mask = tf.reshape(mask, [-1])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=mask)
    loss = tf.reduce_mean(ce)

    return loss
