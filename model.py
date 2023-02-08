import tensorflow as tf
import keras
from keras.layers import *
from keras import Sequential
from keras.losses import MeanSquaredError
import pandas as pd
import numpy as np
from PIL import Image
#from keras import preprocessing
#from preprocessing import image.ImageDataGenerator

# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
# https://medium.com/analytics-vidhya/fastai-image-regression-age-prediction-based-on-image-68294d34f2ed


# https://stackoverflow.com/questions/59568143/neural-network-with-float-labels
# https://www.tensorflow.org/tutorials/keras/regression
# https://keras.io/api/data_loading/
# https://stackoverflow.com/questions/62877412/how-to-load-image-data-for-regression-with-keras
# load the dataset

data_dir = r"./data/images/"
# Neither absolute nor relative path detects images in the directory

csv = pd.read_csv("data/log.csv")

# get only labels
angles_ = csv.get(["angle"])
angles = np.array(angles_['angle'].tolist())
# print(labels)

centers_ = csv.get(["center"])
centers = centers_['center'].tolist()
# get dataframe with image path and angle
df = csv.loc[:,['center','angle']]
# print(df)
# has shape (N, H*W*3) for N images
images = np.array([np.array(Image.open(f"data/images/{f}")).ravel() for f in centers])
#dataset = tf.keras.utils.image_dataset_from_directory(
#    directory = data_dir,
#    labels = angles,
#    image_size=(160, 320))
# ValueError: Expected the lengths of `labels` to match the number of files in the target directory.
# len(labels) is 3404 while we found 0 files in directory data/images/.
print(dataset)
#dataset = tf.keras.preprocessing.image_dataset_from_directory(
#    data_dir, image_size=(320, 160), batch_size=64
#)

#DOES NOT WORK YET
#train_data_gen = image_generator.flow_from_dataframe(dataframe=train_df,
#                                                     x_col='filename',
#                                                     y_col='regression_val',
#                                                     batch_size=BATCH_SIZE,
#                                                     shuffle=True,
#                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))
# Need assistance loading data:
# Deleted left and Right images to simplify the problem
# Has labels loaded from CSV file
# Cannot load images and labels imto a datagenerator 

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

print(model.summary())

model.fit(x_train, y_train, epochs=100, batch_size=20)

accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# ADD PREDICTING Values

