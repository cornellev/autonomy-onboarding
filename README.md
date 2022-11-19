# Autonomy Onboarding Project

Shared repository for the onboarding project for new members of the CEV autonomy subteam. The goal of this project is to predict a steering angle given a single camera input image (even though we're given camera inputs from left, center, and right cameras). Ideally, it would be able to happen in real time :0

# Project Plan

- Read paper “End to End Learning for Self-Driving Cars”
- Set up PyTorch [dataset classes](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and stuff with the given data
- Data augmentation - blur/crop/shadows(changing light levels)/add noise/rotate(change label for steering angle)
- CNN (Pytorch)

## Data

The dataset can be downloaded from [this Kaggle notebook](https://www.kaggle.com/datasets/tusharcode/selfdriving-car-udacity), and has a lot of images with associated direction labels. The images seem to be frames excerpted from a video, and a different analysis that Adams showed us revealed that the vast majority of the directions are pointing straight forward, which will need to be taken into account when dividing the images into training, validation, and testing sets.

The data is stored in the `data/` directory, with images in `data/images/` and a `.csv` file linking each image with a direction number value `data/labels.csv`. The labels were downloaded from the above Kaggle notebook, but had absolute paths to the image files that needed to be cleaned up.

## Helpful Commands

```bash
# update environment after updating `environment.yml`
conda env update --name onboarding --file environment.yml --prune
```
