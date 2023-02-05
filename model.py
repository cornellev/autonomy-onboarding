import tensorflow as tf
import keras
from keras.layers import *
from keras import Sequential
from keras.losses import MeanSquaredError
#import pandas as pd
from numpy import loadtxt
import sys

# https://stackoverflow.com/questions/59568143/neural-network-with-float-labels
# https://www.tensorflow.org/tutorials/keras/regression
# https://keras.io/api/data_loading/
# https://stackoverflow.com/questions/62877412/how-to-load-image-data-for-regression-with-keras
# load the dataset
#data_dir = "/Users/eric/cev/autonomy-onboarding-2022/data/images"
data_dir = "data/images"
# Neither absolute nor relative path detects images in the directory

#raw_dataset = pd.read_csv(data_dir, names=column_names,
#                          na_values='?', comment='\t',
#                          sep=' ', skipinitialspace=True)

#dataset = loadtxt('ZZZZZZZ.csv', delimiter=',') #FILL IN NAME HERE

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, image_size=(320, 160), batch_size=64
)

# Need assistance loading data
# Issue: Center, Right, and Left images are all in 1 folder 
# and not labeled with steering angle
# I cannot figure out a way to load the data with the labels
# Option 1:
# The CSV only has the paths to the images and angle
# Most functions want to load images from a series of files in labeled folders
# Or they want the images themselves labeled
# Solution 1:
# Label the data by putting it into folders (reduces accuracy)
# Solution 2:
# Find a way to label the data with each file name

# Option 2:
# Load the data from a CSV file
# We dont have any of the image data in the CSV file
#
# Solution:
# I'm not sure


# Make training/testing data
x_train = dataset[:,:] # ADD
y_train = dataset[:,:] # CORRECT
x_test = dataset[:,:] # RANGES
y_test = dataset[:,:] # HERE

print(x_train.shape)
print(y_train.shape)
 
print(x_test.shape)
print(y_test.shape)

DenseLayerSize = 320*160*3

# MAKING MODEL WOO
model = Sequential()
model.add(keras.Input(shape=(320,160,3)))
model.add(Flatten())
model.add(Dense(DenseLayerSize, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(1))

# compile model
model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['accuracy']) #asdg

model.fit(x_train, y_train, epochs=100, batch_size=20)

accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# ADD PREDICTING Values

