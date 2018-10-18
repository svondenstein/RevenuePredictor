#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf
import numpy as np

def iou(prediction, mask, name):
    # Compute the argmax for the output to match mask shape
    prediction = tf.expand_dims(tf.argmax(prediction, axis=-1), axis=-1)
    # Thresholds as specified by competition
    thresholds = np.arange(0.5, 0.95, 0.05, dtype='float32')
    # Compute precion at thresholds
    precision = tf.metrics.precision_at_thresholds(mask, prediction, thresholds)
    # Compute the mean for this step
    mean_iou = tf.reduce_mean(precision, name=name)
    
    return mean_iou

def cross_entropy(prediction, mask, num_classes, name):
    # Encode mask as one_hot to match prediction
    mask = tf.one_hot(tf.squeeze(mask), num_classes, dtype=tf.float32)
    # Compute binary cross_entropy
    loss = tf.keras.backend.binary_crossentropy(mask, prediction)
    # Compute the mean for this step
    loss = tf.reduce_mean(loss, name=name)

    return loss
