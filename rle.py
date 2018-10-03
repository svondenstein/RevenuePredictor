#
# Stephen Vondenstein
# 10/03/2018
#

import argparse
import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", default="data/output")

def RLenc(img, order='F', format=True):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def main():
    FLAGS = parser.parse_args()
    pred_ids = next(os.walk(FLAGS.source_folder))[2]
    preds = np.zeros((len(pred_ids), 101, 101), dtype=np.uint8)
    
    print("Resizing " + str(len(pred_ids)) + " images...")
    for i, id_ in enumerate(pred_ids):
        path = FLAGS.source_folder
        img = image.load_img(path + '/' +  id_)
        x = image.img_to_array(img)[:,:,1]
        x = resize(x, (101, 101), mode='constant', preserve_range=True)
        preds[i] = x
        if i % (len(pred_ids)/5) == 0 and i != 0:
            print("Resized " + str(i) + " images [" + str(100*i/len(pred_ids)) + "%]")
    print("Resized " + str(len(pred_ids)) + " images [100.0%]")

    print("Computing RLE of " + str(len(pred_ids)) + " images...")
    pred_dict = {}
    for i, fn in enumerate(pred_ids):
        pred_dict_val = {fn[:-4]:RLenc(np.round(preds[i]))}
        pred_dict.update(pred_dict_val)
        if i % (len(pred_ids)/5) == 0 and i != 0:
            print("Computed RLE for " + str(i) + " images [" + str(100*i/len(pred_ids)) + "%]")
    print("Computed RLE for " + str(len(pred_ids)) + " images [100.0%]")

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission.csv')
    print("Submission saved to ./submission.csv")

if __name__ == "__main__":
    main()
