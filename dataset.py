import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

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


def flip_image(original):
    print("testing")
    # fig = plt.figure()
    # plt.subplot(1,2,1)
    # plt.title('Original image')
    # plt.imshow(original)

    # plt.subplot(1,2,2)
    # plt.title('Original image')

    flipped = tf.image.flip_left_right(original)
    visualize(original, flipped)

def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  plt.show()

def main():
    fig = plt.figure()
    plt.subplot(1,3,1)
    center = mpimg.imread('data/images/center_2022_04_10_12_44_27_913.jpg')
    plt.title('Center')
    plt.imshow(center)
    plt.subplot(1,3,2)
    left = mpimg.imread('data/images/left_2022_04_10_12_44_27_913.jpg')
    plt.title('Left')
    plt.imshow(left)
    plt.subplot(1,3,3)
    right = mpimg.imread('data/images/right_2022_04_10_12_44_27_913.jpg')
    plt.title('Right')
    plt.imshow(right)
    plt.show()


    labels = pd.read_csv("data/log.csv")
    #plot_histogram(df['angle'], 10)
    img = mpimg.imread('data/images/center_2022_04_10_12_44_27_913.jpg')
    flip_image(img)


if __name__ == "__main__":
    main()