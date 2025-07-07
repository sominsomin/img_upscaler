# About

A simple ImageUpscaler built with pytorch. This comes an installable package with a bundled trained model to upscale by 2x. 
There are also scripts related to training in [training/](training/).
I have used an open source dataset from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/) for the training.

# Installation

This repo can be installed as a package:
`pip install .`

# Upscale an image

After the package is installed you can run:
`image_upscaler_cnn input_image.png target_file.png`
to upscale an image by factor 2x.
The model was trained on 500 by 500 pixel images and performs well on similar input sizes. The model though is agnostic of a specific input size for an image.

# Data

Data can be downloaded here:
[Dataset source](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Extract into the data/ folder.

# Training

For just the training, the package doesn't need to be installed, you can just run the scripts.

## Installation

`pip install -r requirements.txt`

## Train

Scripts related to training can be found in [training/](training/).

train a model: [train.py](training/train.py)
test a trained model on a validation dataset (prints the errors): [test_model.py](training/test_model.py)
test a trained model on some example input image from the training dataset: [test_single_image.py](training/test_single_image.py)

the settings for training are set here: [settings.py](training/settings.py)
````
INPUT_WIDTH = 1000
INPUT_HEIGHT = 1000
SCALING_FACTOR = 2
````
Changing the SCALING_FACTOR will train a model with a larger scaling factor. Larger Scaling Factor means a longer training time.
By changing INPUT_WIDTH, INPUT_HEIGHT you can change the size of the input images for training. By this you can reduce the image size of your training data, to speed up training or target a specific size in which your model should perform well. The input size of the training images has a strong impact on the convolutional kernels, so if you train it on (500,500) images, it might not generalize so well to (1000,1000) or larger images. The model though is agnostic of input size, so it works on any input size image.
Just make sure that INPUT_WIDTH * SCALING_FACTOR and INPUT_HEIGHT * SCALING_FACTOR are smaller or equal to the training data images.