import numpy as np
import matplotlib.pyplot as plt

def show_angle(image, angle, color=(1, 0, 0)):
    """ Given an input image (as a numpy array) and associated steering angle (between -1 and 1), returns a new image with a vector plotted on it representing the associated steering angle. Uses color, which is optional and should be an array-like of length 3."""
    # angle is -1 to 1 and gets mapped to
    # -25 to 25 in the simulation, FACING UP
    angle = angle * 25
    length = 50

    #plot steering angle vector
    h, w = np.shape(image)[0:2]
    u = length * np.cos(np.deg2rad(90 - angle))
    v = length * -np.sin(np.deg2rad(90 - angle))

    t = np.linspace(0, 1, 100)
    x = (w / 2) + u * t
    y = (h) + v * t
    points = np.stack((y, x)).T
    points = points.astype(int)
    points = np.unique(points, axis=0)
    # correct for linspace bounds
    points = points[points.T[0] < h - 1]
    points = points[points.T[1] < w - 1]
    # make the line
    tmp = np.array(image)
    tmp[tuple(points.T)] = color
    # make the line thicker
    tmp[tuple((points + [1, 0]).T)] = color
    tmp[tuple((points + [0, 1]).T)] = color
    # place tmp where tmp is non-empty
    return tmp


def show_2_angles(image, angle_pred, angle_truth):
    """ Given an input image (as a numpy array) and two steering angles (between -1 and 1), returns a new image with 2 vectors plotted on it representing both steering angles. """
    # color prediction in red
    tmp = show_angle(image, angle_pred, color=(1, 0, 0))
    # color truth in green
    tmp = show_angle(tmp, angle_truth, color=(0, 1, 0))
    return tmp


if __name__ == "__main__":
    from PIL import Image

    data = np.genfromtxt("data/log.csv",
        delimiter=",",
        dtype=str)
    # test value
    row = data[658]
    filename = f"data/images/{row[0]}"
    image = np.array(Image.open(filename)) / 255.
    angle = float(row[-2])

    visual = show_2_angles(image, np.random.random()*2-1, angle)
    Image.fromarray((visual * 255).astype(np.uint8)).show()
