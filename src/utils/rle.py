#
# Stephen Vondenstein
# 10/03/2018
#
import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
from tqdm import tqdm
from src.utils.utility import get_max_unused_filename


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


def prepare_submission(source_dir, output_path, sub_prefix):
    pred_ids = next(os.walk(source_dir))[2]
    preds = []

    tt = tqdm(range(len(pred_ids)), total=len(pred_ids),
              desc="Loading ")

    for t in tt:
        img = image.load_img(source_dir + '/' + pred_ids[t])
        x = image.img_to_array(img)[:, :, 1]
        preds.append(x)

    tt.close()

    tt = tqdm(range(len(pred_ids)), total=len(pred_ids),
              desc="Computing RLE ")

    pred_dict = {}
    for i, fn in enumerate(pred_ids):
        pred_dict_val = {fn[:-4]: rle(np.round(preds[i]))}
        pred_dict.update(pred_dict_val)
        tt.update()

    tt.close()

    print('Preparing submission...')
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']

    i = 1
    while os.path.exists(os.path.join(output_path, '%s-%s.csv' % (sub_prefix, i))):
        i += 1
    submission_path = get_max_unused_filename(output_path, sub_prefix, '.csv')
    sub.to_csv(submission_path)

    print('Submission saved to ' + submission_path)

    return submission_path

