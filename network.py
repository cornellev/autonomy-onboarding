import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms 
from torch.utils import data  

#https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
#https://medium.com/analytics-vidhya/complete-guide-to-build-cnn-in-pytorch-and-keras-abc9ed8b8160

class TurnNet(nn.Module):

    def __init__(self):
        super(TurnNet, self).__init__()

        self.model = nn.Sequential(
            # convolve and ReLU (values from NVIDIA paper)
            # todo: the first couple should be strided
            nn.Conv2d(3, 24, 5),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            # flatten the data
            nn.Flatten(),
            # dense part
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU())


    def forward(self, x):
        return self.model(x)