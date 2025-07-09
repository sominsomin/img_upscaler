# About

A simple ImageUpscaler built with pytorch. This comes as an installable package with a pretrained model to upscale by 2x. 

There are also scripts related to training in [training/](training/).

I have used an open source dataset from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/) for the training.

# Installation

You can install this project as a Python package directly from the repository:

````
pip install .
````

Alternatively, to set up the environment for training, first clone the repository:

````
git clone https://github.com/sominsomin/img_upscaler.git
cd img_upscaler
pip install -r requirements.txt
````

# Usage: Upscale an Image

Once the image-upscaler package is installed, you can upscale an image from your command line:
````
image_upscaler_cnn input_image.png output_image_2x.png
````
Replace `input_image.png` with the path to your input image and `output_image_2x.png` with your desired output file name.

A note on performance:
The bundled model was trained on (500,500) pixel images. While the model architecture is agnostic to specific input image sizes, its performance (generalization) is best on images around this training dimension. For larger or smaller images, retraining the model on more diverse or representative data might yield better results.

# Approach

This approach uses 3 convolutional layers and an upscaling layer to do image upscaling. The model architecture is agnostic of a specific image size, but the upscaling factor is a constant set during training.

Reference:
[Image Super-Resolution Using Deep Convolutional Networks by Dong et al.](https://arxiv.org/abs/1501.00092).

# Training

For just the training, the package doesn't need to be installed, you can just run the scripts.

## Data

Data can be downloaded here:
[Dataset source](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Extract into the data/ folder if you want to do your own training with custom data. You will also have to set the proper input folders in the training script.

## Installation

````
pip install -r requirements.txt
````

## Train

Scripts related to training can be found in [training/](training/).

- train a model: [training/train.py](training/train.py).
- test a trained model on a validation dataset: [training/test_model.py](training/test_model.py).
- test a trained model on some example input image from the training dataset, outputs the upscaled image: [training/test_single_image.py](training/test_single_image.py).

## Configuration

the settings for training are set here: [training/settings.py](training/settings.py)
````
INPUT_WIDTH = 500
INPUT_HEIGHT = 500
SCALING_FACTOR = 2
````
- SCALING_FACTOR: Change this to train a model with a larger upscaling factor (e.g., 3, 4). Larger factors typically require longer training times and more computational resources.
- INPUT_WIDTH, INPUT_HEIGHT: change the input size of images used during training.
    - Impact on Generalization: Training on specific input dimensions strongly influences the learned convolutional kernels. For example, a model trained exclusively on 500times500 images might not generalize optimally to 1000times1000 or larger images, even though the model itself can technically process any input size.
    - Performance vs. Accuracy: Reducing the input image size for training can speed up the process, or allow you to target specific sizes where the model should perform best.
    - Data Compatibility: Ensure that INPUT_WIDTH * SCALING_FACTOR and INPUT_HEIGHT * SCALING_FACTOR are less than or equal to the dimensions of your actual training data images.