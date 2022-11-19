import torch, torch.utils.data
import pandas as pd
import numpy as np
from PIL import Image
import os

class CarDrivingDataset(torch.utils.data.Dataset):
    """ Dataset for car steering angle. Assumes that the data is in a folder data/ relative to the python root. """
    
    labels: pd.DataFrame

    def __init__(self, transform=None) -> None:
        labels = pd.read_csv("./data/labels.csv")
        # treat each camera input as a separate row
        self.labels = pd.concat(
            labels.get([i, "steering angle"]).rename({ i: "image" }, axis=1)
            for i in ("left image", "center image", "right image")) 

        self.transform = transform


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
        result["image"] = np.moveaxis(image / 255, 2, 0)
            
        # only actually get the steering angle value
        angle = item.loc["steering angle"]
        result["angle"] = angle

        if self.transform:
            result = self.transform(result)

        return result


    def __len__(self):
        return len(self.labels)