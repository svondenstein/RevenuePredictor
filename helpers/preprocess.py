#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf
import math
import random

def process(dataset, training, config, len):
    dataset = dataset.map(parse_data, num_parallel_calls=config.batch_size)
    dataset = dataset.map(normalize_data, num_parallel_calls=config.batch_size)
    if config.augment and training:
        dataset = dataset.concatenate(augment_data(dataset, config))
        dataset = dataset.shuffle(len * 4)
    dataset = dataset.batch(config.batch_size)

    return dataset


def parse_data(image_paths, label_paths, names):
    image_content = tf.read_file(image_paths)
    images = tf.image.decode_png(image_content, channels=1)

    mask_content = tf.read_file(label_paths)
    masks = tf.image.decode_png(mask_content, channels=1)

    return images, masks, names


def resize_data(image, mask, name):
    image = tf.image.resize_images(image, [128, 128])
    mask = tf.image.resize_images(mask, [128, 128])

    return image, mask, name


def normalize_data(image, mask, name):
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = mask / 255.0

    return image, mask, name


def augment_data(dataset, config):
    flip = dataset.map(flip_data, num_parallel_calls=config.batch_size)
    dataset = dataset.concatenate(flip)
    stretch = dataset.map(stretch_data, num_parallel_calls=config.batch_size)
    return flip.concatenate(stretch)


# Tiles the image with flipped tiles and then crops
def tile_image(image):
    batched = (len(image.shape) == 4) #If already batched
    top_flip = tf.image.flip_up_down(image)
    tall_image = tf.concat([top_flip,image,top_flip], batched)
    tall_flipped = tf.image.flip_left_right(tall_image)
    complete_tiled = tf.concat([tall_flipped,tall_image,tall_flipped], batched+1)
    corner = math.ceil((3*101-128)/2)  # Just for clarity
    final = tf.image.crop_to_bounding_box(complete_tiled, corner, corner, 128, 128)
    return final


def tile_data(image, mask, name):
    image_out = tile_image(image)
    mask_out = tile_image(mask)
    return image_out, mask_out, name


def flip_data(image, mask, name):
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)
    return image, mask, name

def stretch_data(image, mask, name):
    factor = random.uniform(0, 1)
    stretched_image = tf.image.resize_images(tf.image.central_crop(image, factor), [image.get_shape()[1], image.get_shape()[2]])
    stretched_mask = tf.image.resize_images(tf.image.central_crop(mask, factor), [mask.get_shape()[1], mask.get_shape()[2]])

    return stretched_image, stretched_mask, name
