# Autonomy Onboarding Project

Shared repository for the onboarding project for new members of the CEV autonomy subteam. The goal of this project is to predict a steering angle given a single camera input image (even though we're given camera inputs from left, center, and right cameras). Ideally, it would be able to happen in real time :0

# Project Plan

- [x] Read paper “End to End Learning for Self-Driving Cars”
- [ ] Process the dataset to remove potential bias
- [ ] Try training a low-parameter Multi-Layer Perception
- [ ] Once it works, scale up!

## Setup

Setting up the environment is quite tricky because TensorFlow does not have native support for Apple M1 Chips. The following commands will set up a virtual environment and install the correct packages to run the code.

```bash
# create the virtual environment
python -m venv ./venv
source ./venv/bin/activate
# update pip
python -m pip install -U pip
```

Instructions from [this Apple website](https://developer.apple.com/metal/tensorflow-plugin/) and a [related StackOverflow page](https://stackoverflow.com/questions/74792286/cant-get-tensorflow-working-on-macos-m1-pro-chip/74806936#74806936).

```bash
# install specific versions of M1 TensorFlow
python -m pip install tensorflow-macos==2.9.0
python -m pip install tensorflow-metal==0.5.0
```

The rest of the packages can be installed with `pip install -r requirements.txt` once we save them there.

## Data

The dataset can be downloaded from [this Kaggle notebook](https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning?select=self_driving_car_dataset_jungle), and has a lot of images with associated direction labels. The images seem to be frames excerpted from a video, and a different analysis that Adams showed us revealed that the vast majority of the directions are pointing straight forward, which will need to be taken into account when dividing the images into training, validation, and testing sets.

The data is stored in the `data/` directory, with images in `data/images/` and a `.csv` file linking each image with a direction number value `data/labels.csv`. The labels were downloaded from the above Kaggle notebook, but had absolute paths to the image files that needed to be cleaned up.

## Progress

We got data loading successfully and after much debugging, figured out how to manipulate the shapes so that there weren't any errors when attempting to train. However, due irregularities in the data, the training was very difficult, and an over-fit model just always predicted the zero vector. Not exactly sure why, and we weren't sure what to try next...

So we decided to just try to run the [Keras model already available online](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project). That didn't work either, because we ignored a warning on its README that stated the project wasn't meant to be usable.

After discussing with a Deep Learning nerd, we are going to try focusing more on cleaning the data and also training a simpler and smaller network to see whether it will work faster.
