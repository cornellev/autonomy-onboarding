# Autonomy Onboarding Project

Shared repository for the onboarding project for new members of the CEV autonomy subteam. The goal of this project is to predict a steering angle given a single camera input image (even though we're given camera inputs from left, center, and right cameras). Ideally, it would be able to happen in real time :0

# Project Plan

- [x] Read paper “End to End Learning for Self-Driving Cars”
- [ ] Process the dataset to remove potential bias
- [ ] Try training a low-parameter Multi-Layer Perception
- [ ] Once it works, scale up!

## Data

The dataset can be downloaded from [this Kaggle notebook](https://www.kaggle.com/datasets/tusharcode/selfdriving-car-udacity), and has a lot of images with associated direction labels. The images seem to be frames excerpted from a video, and a different analysis that Adams showed us revealed that the vast majority of the directions are pointing straight forward, which will need to be taken into account when dividing the images into training, validation, and testing sets.

The data is stored in the `data/` directory, with images in `data/images/` and a `.csv` file linking each image with a direction number value `data/labels.csv`. The labels were downloaded from the above Kaggle notebook, but had absolute paths to the image files that needed to be cleaned up.

## Progress

We got data loading successfully and after much debugging, figured out how to manipulate the shapes so that there weren't any errors when attempting to train. However, due irregularities in the data, the training was very difficult, and an over-fit model just always predicted the zero vector. Not exactly sure why, and we weren't sure what to try next...

So we decided to just try to run the [Keras model already available online](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project). That
