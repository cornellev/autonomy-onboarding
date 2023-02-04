import tensorflow as tf
import keras
from keras.layers import *
from keras import Sequential
from keras.losses import MeanSquaredError

from numpy import loadtxt

# load the dataset
dataset = loadtxt('ZZZZZZZ.csv', delimiter=',') #FILL IN NAME HERE

# Make training/testing data
x_train = dataset[:,:] # ADD
y_train = dataset[:,:] # CORRECT
x_test = dataset[:,:] # RANGES
y_test = dataset[:,:] # HERE

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

