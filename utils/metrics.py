#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf
import numpy as np


def iou(prediction, mask, batch_size):
    metric = []
    epsilon = 1e-15
    predictions = tf.unstack(prediction, num=batch_size)
    masks = tf.unstack(mask, num=batch_size)
    for i in range(batch_size):
        s = masks[i]
        t = predictions[i]
        if tf.reduce_sum(s) == 0 and tf.reduce_sum(t) == 0:
            metric.append(1.0)
        elif tf.reduce_sum(s) == 0 and tf.reduce_sum(t) != 0:
            metric.append(0.0)
        else:
            intersection = tf.reduce_sum(tf.multiply(s, t))
            union = tf.subtract(tf.add(tf.reduce_sum(s), tf.reduce_sum(t)), intersection)
            iou = tf.divide(tf.add(intersection, epsilon), tf.add(union, epsilon))
            thresholds = np.arange(0.5, 0.95, 0.05)
            miou = []
            for thresh in thresholds:
                miou.append(tf.cond(iou > thresh, true_fn=lambda: 1, false_fn=lambda: 0))
            metric.append(tf.reduce_mean(miou))

    return tf.reduce_mean(metric)


def cross_entropy(prediction, mask, num_classes):
    mask = tf.cast(mask, tf.int32)
    prediction = tf.reshape(prediction, (-1, num_classes))
    mask = tf.reshape(mask, [-1])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=mask)
    loss = tf.reduce_mean(ce)

    return loss
