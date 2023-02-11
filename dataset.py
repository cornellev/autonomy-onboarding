import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import multiprocessing as mp
import sys

'''
TO DO:
- flip
- shear
- blur
- color changing

- do in-memory
'''

# (320*160*3, n) for n images
IMAGES = []
ANGLES = []

# load image files
csv = pd.read_csv("data/log.csv")
filenames = csv["center"].tolist()
angles = csv["angle"].tolist()

# add entries into dataset; augment conditionally
# TODO shuffle the filenames to screw with stuff
for i, filename in enumerate(filenames):
    angle = angles[i]
    image = mpimg.imread(f'data/images/{filename}')
    image /= 255

    # perform augmentations
    ...

    # save to datasets
    IMAGES.append(np.ravel(image))
    ANGLES.append(angle)

# normalize the data?

def plot_histogram(data, num_bins):
    plt.hist(data['angle'], bins=num_bins)
    plt.show()


def adjust_brightness(image_filename):
    labels = pd.read_csv("data/log.csv", index_col=0)
    #plot_histogram(df['angle'], 10)
    non_zero_labels = labels.loc[labels['angle'] != 0]
    length = len(non_zero_labels)
    angle = non_zero_labels.loc[non_zero_labels['center'] == image_filename]
    angle = angle['angle'].to_list()[0]
    speed = non_zero_labels.loc[non_zero_labels['center'] == image_filename]
    speed = speed['speed'].to_list()[0]

    image_filename = os.path.splitext(image_filename)[0]

    if f'{image_filename}_brighter.jpg' in non_zero_labels['center'].values:
        return
    img = mpimg.imread(f'data/images/{image_filename}.jpg')
    brighter = tf.image.adjust_brightness(img, 0.3)
    plt.imshow(brighter)
    plt.savefig(f'data/images/{image_filename}_brighter.jpg')
    dimmer = tf.image.adjust_brightness(img, -0.3)
    plt.imshow(dimmer)
    plt.savefig(f'data/images/{image_filename}_dimmer.jpg')
    df = pd.DataFrame([[f'{image_filename}_brighter.jpg', None, None, angle, speed], [f'{image_filename}_dimmer.jpg', None, None, angle, speed]], columns=labels.columns)
    return df

def visualize(original, augmented):
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  plt.show()

def compare_datasets(og_labels, labels):
    plt.subplot(1,2,1)
    plt.title('Original Dataset')
    plt.hist(og_labels['angle'], bins=10)
    plt.subplot(1,2,2)
    plt.title('Augmented Dataset')
    plt.hist(labels['angle'], bins=10)
    plt.show()

def main():
    labels = pd.read_csv("data/log.csv", index_col=0)
    og_labels = labels.loc[~labels['center'].str.contains('dimmer')]
    og_labels = og_labels.loc[~og_labels['center'].str.contains('brighter')]
    #compare_datasets(og_labels, labels)
    non_zero_labels = labels.loc[labels['angle'] != 0]
    labels_lst = non_zero_labels.center.to_list()
    with mp.Pool(7) as pool:
        pool = pool.map(adjust_brightness, labels_lst)
        aug_labels = pd.concat(pool, ignore_index=True)
        labels = pd.concat([labels, aug_labels], ignore_index=True)
        labels.to_csv("data/log.csv")
        print(labels)
    labels.to_csv("data/log.csv")


if __name__ == "__main__":
    ...