import torch, torch.utils.data
import pandas as pd
import numpy as np
from PIL import Image
import os

class CarDrivingDataset(torch.utils.data.Dataset):
    """ Dataset for car steering angle. Assumes that the data is in a folder data/ relative to the python root. """
    
    labels: pd.DataFrame

    def __init__(self, transform=None, shuffle=False) -> None:
        labels = pd.read_csv("./data/labels.csv")
        # treat each camera input as a separate row
        self.labels = pd.concat(
            labels.get([i, "steering angle"]).rename({ i: "image" }, axis=1)
            for i in ("left image", "center image", "right image")) 
        if shuffle:
            self.labels = self.labels.sample(frac=1)

        self.transform = transform

    
    def random_distort(img, angle):
        ''' 
        method for adding random distortion to dataset images, including random brightness adjust, and a random
        vertical shift of the horizon position
        '''
        new_img = img.astype(float)
        # random brightness - the mask bit keeps values from going beyond (0,255)
        value = np.random.randint(-28, 28)
        if value > 0:
            mask = (new_img[:,:,0] + value) > 255 
        if value <= 0:
            mask = (new_img[:,:,0] + value) < 0
        new_img[:,:,0] += np.where(mask, 0, value)
        # random shadow - full height, random left/right side, random darkening
        h,w = new_img.shape[0:2]
        mid = np.random.randint(0,w)
        factor = np.random.uniform(0.6,0.8)
        if np.random.rand() > .5:
            new_img[:,0:mid,0] *= factor
        else:
            new_img[:,mid:w,0] *= factor
        # randomly shift horizon
        h,w,_ = new_img.shape
        horizon = 2*h/5
        v_shift = np.random.randint(-h/8,h/8)
        pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
        pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
        return (new_img.astype(np.uint8), angle)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # build result
        result = {}
        item = self.labels.iloc[index]

        file = os.path.join(".", "data", "images", item.loc["image"])
        # normalize the image
        image = np.array(Image.open(file), dtype=np.float32)
        # (i, j, rgb) -> (rgb, i, j)
        # todo: use YUV like the NIVIDIA paper
        result["image"] = image / 255
            
        # only actually get the steering angle value
        angle = item.loc["steering angle"]
        result["angle"] = np.float32(angle)

        if self.transform:
            result["image"] = self.transform(result["image"])

        return result


    def __len__(self):
        return len(self.labels)