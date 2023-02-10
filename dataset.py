import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
import multiprocess as mp

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


def adjust_brightness():
    labels = pd.read_csv("data/log.csv")
    #plot_histogram(df['angle'], 10)
    non_zero_labels = labels.loc[labels['angle'] != 0]
    length = len(non_zero_labels)
    counter = 0
    for image_filename in non_zero_labels.center.to_list():
        angle = non_zero_labels.loc[non_zero_labels['center'] == image_filename]
        angle = angle['angle'].to_list()[0]
        speed = non_zero_labels.loc[non_zero_labels['center'] == image_filename]
        speed = speed['speed'].to_list()[0]

        image_filename = os.path.splitext(image_filename)[0]

        if f'{image_filename}_brighter.jpg' in non_zero_labels['center'].values:
            continue
        img = mpimg.imread(f'data/images/{image_filename}.jpg')
        brighter = tf.image.adjust_brightness(img, 0.3)
        plt.imshow(brighter)
        plt.savefig(f'data/images/{image_filename}_brighter.jpg')
        dimmer = tf.image.adjust_brightness(img, -0.3)
        plt.imshow(dimmer)
        plt.savefig(f'data/images/{image_filename}_dimmer.jpg')
        labels.loc[len(labels.index)] = [f'{image_filename}_brighter.jpg', None, None, angle, speed]
        counter += 1        
        print(f'{counter}/{length}')
    
    pool = mp.Pool() #creates a pool of process, controls worksers
    #the pool.map only accepts one iterable, so use the partial function
    #so that we only need to deal with one variable.
    results = pool.map(processInput, non_zero_labels.center.to_list()) #make our results with a map call
    pool.close() #we are not adding any more processes
    pool.join() #tell it to wait until all threads are done before going on
    labels.to_csv('data/log.csv', index=0)

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

    labels = pd.read_csv("data/log.csv")
    #plot_histogram(df['angle'], 10)
    non_zero_labels = labels.loc[labels['angle'] != 0]
    adjust_brightness()




if __name__ == "__main__":
    main()