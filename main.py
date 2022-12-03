import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor

from network import TurnNet
from dataset import CarDrivingDataset


transform = Compose([ToTensor(), Resize((66, 200))])
# see https://discuss.pytorch.org/t/dictionary-in-dataloader/40448 for proper use
loader = DataLoader(CarDrivingDataset(transform=transform), batch_size=8, shuffle=True)

model = TurnNet()
model.to("cpu")

# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for entry in loader:
    image = entry["image"]
    angle = entry["angle"]

    # note: the NVIDIA paper assumes that the input planes are 3@66x200,
    # but the ones that we have are 3@160@320. That is not the same!

    for i in range(0, image.shape[0]):
        optimizer.zero_grad()
        result = model(image[i])
        # optimize MSE
        loss = criterion(result, angle)
        print(f"predicted {result} with loss {loss}")

        loss.backward()
        optimizer.step()

pass