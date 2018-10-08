Salt Deposit Identification in Python
==================

## About This Program

This program identifies Salt Deposits by analyzing subsurface images. Kaggle Competition Entry.

### Data

This project was trained and tested using the dataset from the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge) Kaggle competition. (If you'd like to use this dataset for commercial purposes, you may have to contact the sponsor of the competition).

The source images and masks in this dataset are greyscale images of size 101 by 101, and the depth information for each image is included in a _.csv_ file. Preprocessing was done in _utility.py_ to resize the images to achieve a useable tensor shape in the model. Postprocessing was applied in _rle.py_ to ensure a smooth mask, and to compute the RLE (run length encoded) masks, which is the encoding required by the competition sponsor.

## Running Locally

- In Progress.

## Licensing

The model design and preprocessing steps in _model.py_ and _utility.py_ are based on the the [Fully Convolutional DenseNet](https://github.com/HasnainRaz/FC-DenseNet-TensorFlow) implementation by GitHub user [HasnainRaz](https://github.com/HasnainRaz). This implementation was very helpful when designing our network structure.
