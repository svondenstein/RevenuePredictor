#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf


def iou(prediction, mask, num_classes):
    #mask = tf.reshape(mask, [-1])
    #prediction = tf.reshape(prediction, [-1])
    inter = tf.reduce_sum(tf.multiply(prediction, mask))
    union = tf.reduce_sum(tf.subtract(tf.add(prediction, mask), tf.multiply(prediction, mask)))
    mean_iou = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))

    return mean_iou

# def cross_entropy(prediction, mask, num_classes):
#     mask = tf.cast(mask, tf.int32)
#     prediction = tf.reshape(prediction, [tf.shape(prediction)[0], -1, num_classes])
#     mask = tf.reshape(mask, [tf.shape(mask)[0], -1])
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=prediction, labels=mask)
#
#     return loss

def cross_entropy(prediction, mask, num_classes):
    mask = tf.cast(mask, tf.int32)
    prediction = tf.reshape(prediction, (-1, num_classes))
    mask = tf.reshape(mask, [-1])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=mask)
    loss = tf.reduce_mean(ce)

    return loss