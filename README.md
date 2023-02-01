# Autonomy Onboarding Project

Shared repository for the onboarding project for new members of the CEV autonomy subteam. The goal of this project is to predict a steering angle given a single camera input image. Implementing that ourselves in PyTorch proved too difficult considering our lack of familiarity with the framework, so this repository just contains the working code from [this repo](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project) updated to work properly in 2023.

The repo connects to a [Unity simulation](https://github.com/udacity/self-driving-car-sim/tree/Unity_2020_3) using `socketio`.

## Setup

Setting up the environment is quite tricky. The original code was written before M1 chips existed, so I had to do some manual TensorFlow/Keras migration to get it to run properly. The following commands will set up a virtual environment and install the correct packages to run the code.

```bash
# create the virtual environment
python -m venv ./venv-metal
source ./venv-metal/bin/activate
# update pip
python -m pip install -U pip
```

Instructions from [this Apple website](https://developer.apple.com/metal/tensorflow-plugin/) and a [related StackOverflow page](https://stackoverflow.com/questions/74792286/cant-get-tensorflow-working-on-macos-m1-pro-chip/74806936#74806936).

```bash
# install specific versions of M1 TensorFlow
python -m pip install tensorflow-macos==2.9.0
python -m pip install tensorflow-metal==0.5.0
```

And then these are the packages just generally required.

```bash
pip install eventlet pillow opencv-python flask scikit-learn matplotlib python-socketio
```

## More Details

The original project was written in an old version of Keras, which has since been upgraded in a backwards-incompatible manner.

After fixing some library porting issues such as `collections.Iterable` no longer being an alias for `collections.abc.Iterable` and the depracation of some old fixes, `drive.py` would refuse to load the model from the original `model.json` (now saved as `model-old.json`).

According to [this GitHub issue](https://github.com/keras-team/keras/issues/7440#issuecomment-321098478), the problem is that Keras uses Python `marshal` internally in order to serialize lambda functions to JSON instead of `pickle`, and `marshal` makes breaking changes to its format even on minor python version changes.

The solution was to look at the original [model source code](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project/blob/master/model.py#L268), rewrite it in terms of the new version of Keras (such as replacing `Convolution2D` with `Conv2D` and adjusting arguments accordingly), and then save _that_ model into `model.json`.

This worked, and the model could initialize itself using the JSON file and the weights in `model.h5`!
