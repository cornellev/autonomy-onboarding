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
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```

Test that it works on your machine by entering a Python terminal and running the script below, which is a modify version of the verify script at the bottom of the Apple website above with a fix from the above StackOverflow link. If it starts training the first epoch, then everything installed correctly.

```python
import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)
```

The rest of the packages can be installed with `pip install -r requirements.txt` once we save them there.

## Data

The dataset can be downloaded from [this Kaggle notebook](https://www.kaggle.com/datasets/andy8744/udacity-self-driving-car-behavioural-cloning?select=self_driving_car_dataset_jungle), and has a lot of images with associated direction labels. The images seem to be frames excerpted from a video, and a different analysis that Adams showed us revealed that the vast majority of the directions are pointing straight forward, which will need to be taken into account when dividing the images into training, validation, and testing sets.

The data is stored in the `data/` directory, with images in `data/images/` and a `.csv` file linking each image with a direction number value `data/labels.csv`. The labels were downloaded from the above Kaggle notebook, but had absolute paths to the image files that needed to be cleaned up.

We are implementing data augmentation in order to reduce the bias toward a 0 degree steering angle for all outputs. The methods we will be attempting are the following
- color changing (black and white, red/blue/green color shift, etc.)
- blur
- shear + crop

We are not augmenting the data by flipping the image, as this will put the center lines of the road on the right side of the driver, which will mess up the steering angle calculations.

## Progress

We got data loading successfully and after much debugging, figured out how to manipulate the shapes so that there weren't any errors when attempting to train. However, due irregularities in the data, the training was very difficult, and an over-fit model just always predicted the zero vector. Not exactly sure why, and we weren't sure what to try next...

So we decided to just try to run the [Keras model already available online](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project). That didn't work either, because we ignored a warning on its README that stated the project wasn't meant to be usable.

After discussing with a Deep Learning nerd, we are going to try focusing more on cleaning the data and also training a simpler and smaller network to see whether it will work faster.
