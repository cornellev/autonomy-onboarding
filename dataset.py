import torch, torch.utils.data
import pandas as pd
import numpy as np
from PIL import Image
import os

class CarDrivingDataset(torch.utils.data.Dataset):
    """ Dataset for car steering angle, throttle, brake, and speed. Assumes that the data is in a folder data/ relative to the python root. """
    
    labels: pd.DataFrame

    def __init__(self, transform=None) -> None:
        self.labels = pd.read_csv("./data/labels.csv")
        self.transform = transform


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # build result
        result = {}
        item = self.labels.iloc[index]
        for key in ("left image", "center image", "right image"):
            file = os.path.join(".", "data", "images", item.loc[key])
            image = np.array(Image.open(file))
            result[key] = image
            
        # only actually get the steering angle value
        angle = item.loc["steering angle"]
        result["angle"] = angle

        if self.transform:
            result = self.transform(result)

        return result


    def __len__(self):
        return len(self.labels)