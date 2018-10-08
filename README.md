Salt Deposit Identification in Python
==================

© 2018 Stephen Vondenstein & Matthew Buckley

## About This Program

This program identifies Salt Deposits by analyzing subsurface images. The network design is based on the FC-DensNet106 network, outlined in the [One Hundred Layers Tiramisu](https://arxiv.org/pdf/1611.09326.pdf) paper. Additional optimizations are made to process depth data and improve masks, which are outlined below.

## Table of Contents

[Data](#data)

[Model](#model)

[Running Locally](#running-locally)
* [Training](#training)
* [Inference](#inference)
* [Submitting to Kaggle](#kaggle-only)

[Argument List](#argument-definitions)

[Licensing](#licensing)

## Data

This project was trained and tested using the dataset from the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge) Kaggle competition. (If you'd like to use this dataset for commercial purposes, you may have to contact the sponsor of the competition).

The source images and masks in this dataset are greyscale images of size 101 by 101, and the depth information for each image is included in a _.csv_ file. Preprocessing was done in _utility.py_ to resize the images to achieve a useable tensor shape in the model. Postprocessing was applied in _rle.py_ to ensure a smooth mask, and to compute the RLE (run length encoded) masks, which is the encoding required by the competition sponsor.

## Model

- In Progress.

## Running Locally

Clone locally via GitHub Desktop or CLI: `git clone https://github.com/svondenstein/salty.git` and follow the directions below to train and predict.

### Training

Using the default arguments:
1. Ensure that directories or symlinks with the following names are in the working directory:
- _checkpoints/_: a directory in which to save checkpoint data for weights and parameters
- _train/_: a directory containing training images and masks. Should contain two subdirectories: _images/_ and _masks/_, containing training images and training masks, respectively.
2. Run the program using the default arguments: `python main.py --mode=train`

To use different training parameters, please consult the argument definitions section of the README.

### Inference

Using the default arguments:
1. Ensure that directories or symlinks with the following names are in the working directory:
- _checkpoints/_: a directory in which to load checkpoint data for weights and parameters
- _test/_: a directory containing training images and masks. Should contain two subdirectories: _images/_ and _masks/_, containing training images and training masks, respectively.
- _predictions/_: an empty directory to contain predicted masks
2. Run the program using the default arguments: `python main.py --mode=infer`

To use different inference parameters, please consult the argument definitions section of the README.

### Kaggle Only - Computing RLE and Assembling Submission

Using the default arguments:
1. Ensure that directories or symlinks with the following names are in the working directory:
- _submissions/_: a directories in which to save the `.csv` files to be submitted
- _predictions/_: a directories containing the predicted masks
2. Run the program using the default arguments: `python rle.py`

To use different submission parameters, please consult the argument definitions section of the README.

## Argument Definitions

_**main.py**_:

| Argument | Definition | Default Value |
| --- | --- | --- |
| `--mode` | Execution mode - may be "train" or "infer" | infer |
| `--train_data` | Path to training data - must contain subdirectories _images/_ and _masks/_ | _./train/_ |
| `--val_data` | Path to validation data - must contain subdirectories _images/_ and _masks/_ | _./train/_ |
| `--ckpt` | Path to directory to store checkpoints & model parameters | _./checkpoints/model.ckpt_ |
| `--layers_per_block` | Number of layers per dense block in the model | 4,5,6,10,12,15 |
| `--batch_size` | Batch size for use in training | 4 |
| `--epochs` | Number of epochs to train | 12 |
| `--num_threads` | Number of threads to use for the data input pipeline | 2 |
| `--growth_k` | Growth rate for Tiramisu | 16 |
| `--num_classes` | Number of classes to predict | 2 |
| `--learning_rate` | Learning rate for optimizer | 1e-4 |
| `--infer_data` | Path to testing data to make predictions for | _./test/_ |
| `--output_folder` | Path to save predicted masks | _./predictions/_ |
| `--prior_model` | Path to model to continue training (optional) | _**none**_ |

_**rle.py**_:

| Argument | Definition | Default Value |
| --- | --- | --- |
| `--source_folder` | Directory containing the predicted masks for which to convert to RLE | ./predictions/ |
| `--image_height` | Target height of mask for RLE computation | 101 |
| `--image_width` | Target width of mask for RLE computation | 101 |
| `--csv` | Path to the output file containing the IDs/RLE values for submission to Kaggle | ./submissions/submission.csv |

## Licensing

The model design and preprocessing steps in _model.py_ and _utility.py_ are based on the the [Fully Convolutional DenseNet Tensorflow](https://github.com/HasnainRaz/FC-DenseNet-TensorFlow) implementation by GitHub user [HasnainRaz](https://github.com/HasnainRaz). This implementation was very helpful when designing our network structure.
