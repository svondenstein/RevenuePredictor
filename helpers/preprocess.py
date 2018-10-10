#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf

def process(dataset, training, config):
    dataset = dataset.map(parse_data, num_parallel_calls=config.batch_size)
    dataset = dataset.map(resize_data, num_parallel_calls=config.batch_size)
    dataset = dataset.map(normalize_data, num_parallel_calls=config.batch_size)

    return dataset

@staticmethod
def parse_data(image_paths, label_paths):
    image_content = tf.read_file(image_paths)
    images = tf.image.decode_png(image_content, channels=1)

    mask_content = tf.read_file(label_paths)
    masks = tf.image.decode_png(mask_content, channels=1)

    return images, masks

@staticmethod
def resize_data(image, mask):
    image = tf.image.resize_images(image, [128, 128])
    mask = tf.image.resize_images(mask, [128, 128])

    return image, mask

@staticmethod
def normalize_data(image, mask):
    image = image / 255.0
    mask = mask / 255.0

    return image, mask
