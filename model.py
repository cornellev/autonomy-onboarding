import tensorflow as tf
import keras
from keras.layers import *
from keras import Sequential
from keras.losses import MeanSquaredError
import pandas as pd
import numpy as np
from PIL import Image


#-----------------load the dataset-----------------
data_dir = r"./data/images/"

#load csv as pandas dataframe(df)
csv = pd.read_csv("data/log.csv")

# get the df of angles & convert to np arr
angles_ = csv.get(["angle"])
angles = np.array(angles_['angle'].tolist())

# get the df of image addresses & convert to np arr
centers_ = csv.get(["center"])
centers = np.array(centers_['center'].tolist())

#N = num images, H = height of image, W = width of image
N = 3404
H = 160
W = 320
image_size = H*W*3
# images arr has shape (N, H*W*3) for N images
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
#-----------------End loading the dataset-----------------
#--------------------MAKING MODEL WOO---------------------
model = Sequential()
model.add(keras.Input(shape=(image_size)))
model.add(Dense(image_size, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(1))

# compile model
model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['accuracy']) #asdg

#prints out a summary of the model layers
print(model.summary())

#Train the model
num_epochs = 100
b_s = 20
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=b_s, verbose = 1)


#evaluate model on test data and print the acc
loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
model.save('Model Saves/V1-' + str(acc*100))
# ADD PREDICTING Values
#y_pred = model.predict(x_test)

