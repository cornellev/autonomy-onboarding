import tensorflow as tf
import keras
from keras.layers import *
from keras import Sequential
from keras.losses import MeanSquaredError
import pandas as pd
import numpy as np
from PIL import Image

from dataset import IMAGES, ANGLES

# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
# https://medium.com/analytics-vidhya/fastai-image-regression-age-prediction-based-on-image-68294d34f2ed
# https://stackoverflow.com/questions/59568143/neural-network-with-float-labels
# https://www.tensorflow.org/tutorials/keras/regression
# https://keras.io/api/data_loading/
# https://stackoverflow.com/questions/62877412/how-to-load-image-data-for-regression-with-keras

#load csv
csv = pd.read_csv("data/log.csv")

# the df of angles & convert to np arr
angles_ = csv.get(["angle"])
angles = np.array(angles_['angle'].tolist())

# the df of image addresses & convert to np arr
centers_ = csv.get(["center"])
centers = np.array(centers_['center'].tolist())
#centers_['center'].tolist() 


# get dataframe with image path and angle
#df = csv.loc[:,['center','angle']]
# print(df)

N = 3404
H = 160
W = 320
# images has shape (N, H*W*3) for N images
images = np.array([np.array(Image.open(f"data/images/{f}")).ravel() for f in centers])
#print(images)

# Make training/testing data
num_images = len(images)
train_test_split = .9

x_train = images[0:int(train_test_split*num_images)]
y_train = centers[0:int(train_test_split*num_images)]
x_test = images[int(train_test_split*num_images):num_images]
y_test = centers[int(train_test_split*num_images):num_images]

# Print shape of train data
print(x_train.shape)
print(y_train.shape)
# Print shape of test data
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
model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['accuracy'])

#prints out a summary of the model layers
print(model.summary())

#Train the model
model.fit(x_train, y_train, epochs=100, batch_size=20)

#evaluate model on test data and print the acc
accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# ADD PREDICTING Values

