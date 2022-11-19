import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms 
from torch.utils import data  

#https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
#https://medium.com/analytics-vidhya/complete-guide-to-build-cnn-in-pytorch-and-keras-abc9ed8b8160

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,24,5)
        self.conv2 = nn.Conv2d(24,36,5)
        self.conv3 = nn.Conv2d(36,48,5)
        self.conv4 = nn.Conv2d(48,64,3)
        self.conv5 = nn.Conv2d(64,64,3)
        #flatten here
        self.fc1 = nn.Linear(1164, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self , x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        output = x
        return output


FILE_NAME = ''
dataset = datasets.ImageFolder(root=FILE_NAME,
transform = transforms.ToTensor())  
loader = data.DataLoader(dataset, batch_size = 8, shuffle = True)

model = Net()
optimizer = optim.Adam(model.parameters())
for (i,l) in trainloader:
    optimizer.zero_grad()
    output = model(i)
    loss = F.nll_loss(output, l)
    loss.backward()
    optimizer.step()