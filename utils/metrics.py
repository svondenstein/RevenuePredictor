#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf
import numpy as np


# TODO: Fix this, I think it's incorrect.
def iou(prediction, mask, batch_size):
    metric = []

    for batch in range(batch_size):
        t = tf.argmax(prediction[batch], axis=2)
        p = tf.cast(tf.squeeze(mask[batch]), tf.int64)
        miou = jaccard(t, p)
        thresholds = np.arange(0.5, 1.0, 0.05)
        s = []
        for thresh in thresholds:
            s.append(tf.cond(miou > thresh, lambda: tf.add(0, 1), lambda: tf.add(0, 0)))
        metric.append(tf.reduce_mean(s))

    return tf.reduce_mean(metric)


def cross_entropy(prediction, mask, num_classes):
    mask = tf.cast(mask, tf.int32)
    prediction = tf.reshape(prediction, (-1, num_classes))
    mask = tf.reshape(mask, [-1])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=mask)
    loss = tf.reduce_mean(ce)

    return loss


def jaccard(prediction, mask):
    epsilon = 1e-15
    intersection = tf.cast(tf.reduce_sum(tf.multiply(prediction, mask)), float)
    union = tf.cast(tf.add(tf.reduce_sum(prediction), tf.reduce_sum(mask)), float)

    return tf.reduce_mean((intersection + epsilon) / (union - intersection + epsilon))
