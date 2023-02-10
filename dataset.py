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
'''



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
    df = pd.DataFrame([[f'{image_filename}_brighter.jpg', None, None, angle, speed], [f'{image_filename}_dimmmer.jpg', None, None, angle, speed]], columns=labels.columns)
    return df

def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  plt.show()

def test(x):
    print("this is in the child process")
    sys.stdout.flush()

def main():
    labels = pd.read_csv("data/log.csv", index_col=0)
    #plot_histogram(df['angle'], 10)
    non_zero_labels = labels.loc[labels['angle'] != 0]
    labels_lst = non_zero_labels.center.to_list()
    with mp.Pool(7) as pool:
        pool = pool.map(adjust_brightness, labels_lst)
        aug_labels = pd.concat(pool, ignore_index=True)
        labels = pd.concat([labels, aug_labels], ignore_index=True)
        labels.to_csv("data/log.csv")
        print(labels)


if __name__ == "__main__":
    main()