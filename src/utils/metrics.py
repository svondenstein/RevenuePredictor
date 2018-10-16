#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf
import numpy as np

def iou(prediction, mask, name):
    metric = []
    elems = (prediction, mask)
    metric.append(tf.reduce_mean(tf.map_fn(mean_iou, elems, dtype=tf.int32)))

    return tf.reduce_mean(metric, name=name)

def mean_iou(elems):
    epsilon = 1e-15
    s = tf.cast(elems[1], tf.float32)
    t = elems[0]
    if tf.reduce_sum(s) == 0 and tf.reduce_sum(t) == 0:
        return 1.0
    elif tf.reduce_sum(s) == 0 and tf.reduce_sum(t) != 0:
        return 0.0
    else:
        intersection = tf.reduce_sum(tf.multiply(s, t))
        union = tf.subtract(tf.add(tf.reduce_sum(s), tf.reduce_sum(t)), intersection)
        iou = tf.divide(tf.add(intersection, epsilon), tf.add(union, epsilon))
        thresholds = np.arange(0.5, 0.95, 0.05)
        miou = []
        for thresh in thresholds:
            miou.append(tf.cond(iou > thresh, true_fn=lambda: 1, false_fn=lambda: 0))
        return tf.reduce_mean(miou)

# def iou(prediction, mask, name):
#     metric = []
#     epsilon = 1e-15
#     for i in range(batch_size):
#         s = mask[i, :, :, :]
#         t = prediction[i, :, :, :]
#         s = tf.cast(s, tf.float32)
#         if tf.reduce_sum(s) == 0 and tf.reduce_sum(t) == 0:
#             metric.append(1.0)
#         elif tf.reduce_sum(s) == 0 and tf.reduce_sum(t) != 0:
#             metric.append(0.0)
#         else:
#             intersection = tf.reduce_sum(tf.multiply(s, t))
#             union = tf.subtract(tf.add(tf.reduce_sum(s), tf.reduce_sum(t)), intersection)
#             iou = tf.divide(tf.add(intersection, epsilon), tf.add(union, epsilon))
#             thresholds = np.arange(0.5, 0.95, 0.05)
#             miou = []
#             for thresh in thresholds:
#                 miou.append(tf.cond(iou > thresh, true_fn=lambda: 1, false_fn=lambda: 0))
#             metric.append(tf.reduce_mean(miou))
#
#     return tf.reduce_mean(metric, name=name)


def cross_entropy(prediction, mask, num_classes, name):
    mask = tf.one_hot(tf.squeeze(mask), num_classes, dtype=tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=prediction)
    loss = tf.reduce_mean(loss, name=name)

    return loss
