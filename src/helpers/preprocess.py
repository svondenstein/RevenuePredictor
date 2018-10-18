#
# Stephen Vondenstein, Matthew Buckley
# 10/09/2018
#
import tensorflow as tf
import math
import random
import numpy as np

def process(dataset, training, config, len):
    dataset = dataset.map(parse_data, num_parallel_calls=config.batch_size)
    dataset = dataset.map(normalize_data, num_parallel_calls=config.batch_size)
    if config.augment and training:
        dataset = dataset.concatenate(augment_data(dataset, config))
        dataset = dataset.shuffle(len * 7)
    #dataset = dataset.map(combine_depth_data, num_parallel_calls=config.batch_size)
    dataset = dataset.batch(config.batch_size)

    return dataset

# Parse images from filenames
def parse_data(image_path, label_path, name, depth):
    image_content = tf.read_file(image_path)
    image = tf.image.decode_png(image_content, channels=1)

    mask_content = tf.read_file(label_path)
    mask = tf.image.decode_png(mask_content, channels=1)

    return image, mask, name, depth

# [DEPRECATED] Resize image to proper size - use tiling now
def resize_data(image, mask, name, depth):
    image = tf.image.resize_images(image, [128, 128])
    mask = tf.image.resize_images(mask, [128, 128])

    return image, mask, name, depth

# Normalize image array between 0-1
def normalize_data(image, mask, name, depth):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    mask = mask // 255
    depth = tf.cast(depth, tf.float32)
    depth = depth / 1000.0

    return image, mask, name, depth

# Add depth data to each image
def combine_depth_data(image, mask, name, depth):
    # depth_layer = tf.fill(tf.shape(image),depth)
    # tf.print(depth_layer)
    # image = tf.concat([image,depth_layer],-1)
    return image, mask, name, depth

# Perform dataset augmentations
def augment_data(dataset, config):
    flip = dataset.map(flip_data, num_parallel_calls=config.batch_size)
    c_flip = dataset.concatenate(flip)
    crop = c_flip.map(crop_data, num_parallel_calls=config.batch_size)
    stretch = c_flip.map(stretch_data, num_parallel_calls=config.batch_size)
    output = c_flip.concatenate(crop)
    output = output.concatenate(stretch)

    return output

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

# [DEPRECATED] Call tile image for image and mask - moved to network
def tile_data(image, mask, name):
    image_out = tile_image(image)
    mask_out = tile_image(mask)
    return image_out, mask_out, name

# Flip images and masks across the vertical axis
def flip_data(image, mask, name, depth):
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)

    return image, mask, name, depth

# Stretch images randomly - similar to crop
# but x and y crop factor are independent
def stretch_data(image, mask, name, depth):
    # Variables used for stretching
    x_factor = int(random.uniform(0.2, 0.8) * 101)
    y_factor = int(random.uniform(0.2, 0.8) * 101)
    seed = np.random.randint(1234)
    # Perform random crop
    stretch_image = tf.random_crop(image, size=[x_factor, y_factor, 1], seed=seed)
    stretch_mask = tf.random_crop(mask, size=[x_factor, y_factor, 1], seed=seed)
    # Resize back to size expected by the network - the input to this operation
    # has a different aspect ratio than the output, so the image gets stretched
    # to fit
    stretch_image = tf.image.resize_images(stretch_image, size=[tf.shape(image)[0],
                                                        tf.shape(image)[1]])
    stretch_mask = tf.image.resize_images(stretch_mask, size=[tf.shape(mask)[0],
                                                        tf.shape(mask)[1]],
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return stretch_image, stretch_mask, name, depth

# Random crop (zoom)
def crop_data(image, mask, name, depth):
    # Variables used for cropping
    factor = int(random.uniform(0.2, 0.8) * 101)
    seed = np.random.randint(1234)
    # Perform random crop
    crop_image = tf.random_crop(image, size=[factor, factor, 1], seed=seed)
    crop_mask = tf.random_crop(mask, size=[factor, factor, 1], seed=seed)
    # Resize back to size expected by the network
    crop_image = tf.image.resize_images(crop_image, size=[tf.shape(image)[0],
                                                        tf.shape(image)[1]])
    crop_mask = tf.image.resize_images(crop_mask, size=[tf.shape(mask)[0],
                                                        tf.shape(mask)[1]],
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return crop_image, crop_mask, name, depth
