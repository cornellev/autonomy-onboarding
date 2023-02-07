import numpy as np
import matplotlib.pyplot as plt

def show_angle(image, angle):
    """ Given an input image (as a numpy array) and associated steering angle (between -1 and 1)
    
    Returns a new image with a vector plotted on it representing the associated steering angle."""
    
    #create figure and axis objects, plot image
    fig, ax = plt.subplots(1,figsize=(10,10))
    ax.imshow(image, cmap='gray')


    #is the conversion correct and does this length work?
    angle = angle * 180
    length = 50

    #plot steering angle vector
    ax.quiver(image.shape[0]/2, image.shape[1]/2, length*np.cos(np.deg2rad(angle)), -(length)*np.sin(np.deg2rad(angle)), angles='xy', scale_units='xy', scale=1, color='red')
    
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    
    #draws image and vector
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    
    #returns image with plotted vector
    return image


def show_2_angles(image, angle1, angle2):
    """ Given an input image (as a numpy array) and two steering angles (between -1 and 1).
    
    Returns a new image with 2 vectors plotted on it representing both steering angles."""

    #create figure with size (10,10) and axis 
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    #is the conversion correct and does this length work?
    angle1= angle1*180
    angle2= angle2*180
    length = 50

    #plot vectors
    ax.quiver(image.shape[0]/2, image.shape[1]/2, length*np.cos(np.deg2rad(angle1)), -(length)*np.sin(np.deg2rad(angle1)), color='red', angles='xy', scale_units='xy', scale=1, headwidth=5)
    ax.quiver(image.shape[0]/2, image.shape[1]/2, length*np.cos(np.deg2rad(angle2)), -(length)*np.sin(np.deg2rad(angle2)), color='blue', angles='xy', scale_units='xy', scale=1, headwidth=5)
    
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])

    #draws image and vectors
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)

    #return image with plotted vectors
    return image


if __name__ == "__main__":
    from PIL import Image

    data = np.genfromtxt("data/log.csv",
        delimiter=",",
        dtype=str)
    # test value
    row = data[658]
    filename = f"data/images/{row[0]}"
    image = np.array(Image.open(filename)) / 255.
    angle = row[-2]

    visual = show_angle(image, angle)
    
    Image.fromarray((image * 255).astype(np.uint8)).show()
    print(angle)
    ...
