#!/usr/bin/python3
#
# Stephen Vondenstein, Matthew Buckley
# 10/14/2018
#
import os
# Add parent to project root so we can import project files
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import math
import cv2

from tqdm import tqdm
from src.utils.parser import get_args
from src.utils.utility import get_max_filename
from src.utils.utility import create_dirs

def csv_to_image(csv):
    images = []
    names = []
    length = tqdm(range(len(csv)), total=len(csv), desc="Converting ")
    for i in length:
        name = csv.loc[i, 'id']
        rle = csv.loc[i, 'rle_mask']
        rle = str(rle).split()
        image = rle_to_image(rle)
        images.append(image)
        names.append(name)
    length.close()

    return images, names

def rle_to_image(rle):
    placeholder = np.zeros(101 * 101)
    if len(rle) == 1:
        image = np.reshape(placeholder, [101, 101]).T
    else:
        start = rle[::2]
        length = rle[1::2]
        for i, v in zip(start, length):
            placeholder[(int(i) - 1):(int(i) - 1) + int(v)] = 255
        image = np.reshape(placeholder, [101, 101]).T
    return image

def mean_iou(source, test):
    ious = []
    length = tqdm(range(len(source)), total=len(source), desc="Computing IoU A")
    for i in length:
        iou = calculate_iou(source[i], test[i])
        ious.append(iou)
    return np.mean(ious)

def vector_iou(source, test):
    length = tqdm(range(len(source)), total=len(source), desc="Computing IoU B ")
    metric = []
    epsilon = 1e-15
    for i in length:
        s = source[i] / 255
        t = test[i] / 255
        if s.sum() == 0 and t.sum() == 0:
            metric.append(1.0)
        elif s.sum() == 0 and t.sum() != 0:
            metric.append(0.0)
        else:
            intersection = (s * t).sum()
            union = s.sum() + t.sum() - intersection
            iou = ((intersection + epsilon) / (union + epsilon))
            thresholds = np.arange(0.5, 0.95, 0.05)
            miou = []
            for thresh in thresholds:
                miou.append(iou > thresh)
            metric.append(np.mean(miou))

    return np.mean(metric)

def calculate_iou(source, test):
    if source.sum() == 0:
        if test.sum() == 0:
            iou = 1.0
        else:
            iou = 0.0
    else:
        union = np.logical_or(source, test).astype('int')
        intersection = np.logical_and(source, test).astype('int')
        iou = intersection.sum() / union.sum()
        iou = math.ceil((max(iou - 0.5, 0))/0.05)/10
    return iou

def save_images(images, names, path):
    length = tqdm(range(len(images)), total=len(images), desc="Saving ")
    for i in length:
        cv2.imwrite(os.path.join(path, names[i] + '.png'), images[i])
    length.close()


def compare_iou(config):
    input_path = config.submission_path
    # Load test images
    test_path = get_max_filename(input_path, 'test', '.csv')
    test = pd.read_csv(test_path)
    test_images, test_image_names = csv_to_image(test)
    if config.compute_iou:
        # Load source images
        source = pd.read_csv('./data/train.csv')
        source_images, _ = csv_to_image(source)
        # Compute IoU values
        iou_a = mean_iou(source_images, test_images)
        iou_b = vector_iou(source_images, test_images)
        print('Mean IoU A: ' + str(iou_a))
        print('Mean IoU B: ' + str(iou_b))
    if config.rle_to_image:
        create_dirs(['./converted_images/'])
        save_images(test_images, test_image_names, './converted_images/')


if __name__ == '__main__':
    # If executing from this folder, change the working directory to the parent to manipulate project files
    if os.path.dirname(os.path.abspath(__file__)) == os.getcwd():
        project_root = os.path.abspath('..')
        os.chdir(project_root)
    # Get args and pass them to compare_rle
    config = get_args()
    compare_iou(config)
