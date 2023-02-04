import numpy as np

def show_angle(image, angle):
    """ Given an input image (as a numpy array) and associated steering angle (between -1 and 1), creates a new image where a human can visibly tell approximately what the associated steering angle should be just by looking at it. """
    # replace this
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