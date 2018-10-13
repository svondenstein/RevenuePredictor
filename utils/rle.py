#
# Stephen Vondenstein
# 10/03/2018
#
import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
from skimage.transform import resize


def rle(img, order='F', format=True):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
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


def prepare_submission(source_dir, output_path):
    pred_ids = next(os.walk(source_dir))[2]
    preds = []
    
    print("Loading " + str(len(pred_ids)) + " images...")
    for i, id_ in enumerate(pred_ids):
        img = image.load_img(source_dir + '/' + id_)
        x = image.img_to_array(img)[:, :, 1]
        preds[i] = x

    print("Computing RLE of " + str(len(pred_ids)) + " images...")
    pred_dict = {}
    for i, fn in enumerate(pred_ids):
        pred_dict_val = {fn[:-4]: rle(np.round(preds[i]))}
        pred_dict.update(pred_dict_val)
        if i % (len(pred_ids)/5) == 0 and i != 0:
            print("Computed RLE for " + str(i) + " images [" + str(100*i/len(pred_ids)) + "%]")
    print("Computed RLE for " + str(len(pred_ids)) + " images [100.0%]")

    print('Preparing submission...')
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(output_path + 'submission.csv')
    print('Submission saved to ' + output_path + 'submission.csv')
