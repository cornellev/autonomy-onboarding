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

# add entries into dataset; augment conditionally
# TODO shuffle the filenames to screw with stuff
#for i, filename in enumerate(filenames):

def populate_dataset(tup: tuple): # tup should be in form (filename, angle)
    filename, angle = tup
    image = mpimg.imread(f'data/images/{filename}')
    return ([np.ravel(image/255)], [angle])

def augment_brightness(tup: tuple): # tup should be in form (filename, angle)
    filename, angle = tup
    image = mpimg.imread(f'data/images/{filename}')

    # perform augmentations
    brighter = tf.image.adjust_brightness(image, 0.3)
    dimmer = tf.image.adjust_brightness(image, -0.3)

    # save to datasets
    return ([np.ravel(brighter/255), np.ravel(dimmer/255)], [angle, angle])

def augment_saturation(tup: tuple): # tup should be in form (filename, angle)
    filename, angle = tup
    image = mpimg.imread(f'data/images/{filename}')

    # perform augmentations
    sat = tf.image.adjust_brightness(image, 4)
    usat = tf.image.adjust_brightness(image, -4)

    # save to datasets
    return ([np.ravel(sat/255), np.ravel(usat/255)], [angle, angle])
    
# load image files
def load_data():
    IMAGES_t = []
    ANGLES_t = []
    csv = pd.read_csv("data/log.csv")
    filenames = csv["center"].tolist()
    angles = csv["angle"].tolist()
    csv = csv.loc[csv['angle'] != 0]
    nonzero_filenames = csv["center"].tolist()
    nonzero_angles = csv["angle"].tolist()
    
    #load initial
    with mp.Pool(7) as pool:
        results = pool.map(populate_dataset, zip(filenames, angles))
        images = np.array([x[0] for x in results])
        angles = np.array([x[1] for x in results])
        IMAGES_t.extend(np.reshape(images, (images.shape[0]*images.shape[1], -1)))
        ANGLES_t.extend(np.ravel(angles))
    for aug_func in [augment_brightness, augment_saturation]:
        with mp.Pool(7) as pool:
            results = pool.map(aug_func, zip(nonzero_filenames, nonzero_angles))
            images = np.array([x[0] for x in results])
            angles = np.array([x[1] for x in results])
            IMAGES_t.extend(np.reshape(images, (images.shape[0]*images.shape[1], -1)))
            ANGLES_t.extend(np.ravel(angles))
    IMAGES[:] = IMAGES_t
    ANGLES[:] = ANGLES_t
    return IMAGES, ANGLES

# normalize the data?





def compare_images(image, brighter, dimmer):
    plt.subplot(1,3,1)
    plt.imshow(brighter)
    plt.subplot(1,3,2)
    plt.imshow(image)
    plt.subplot(1,3,3)
    plt.imshow(dimmer)
    plt.show()

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
    load_data()