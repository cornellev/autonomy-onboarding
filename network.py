import torch
import torch.nn as nn
import torch.nn.functional as F

#https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
#https://medium.com/analytics-vidhya/complete-guide-to-build-cnn-in-pytorch-and-keras-abc9ed8b8160
class TurnNet(nn.Module):

    def __init__(self):
        super(TurnNet, self).__init__()
        # https://www.baeldung.com/cs/cnn-depth
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        #flatten here
        self.fc1 = nn.Linear(18 * 64, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)


    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))
        x = torch.flatten(x, 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        output = x
        return output