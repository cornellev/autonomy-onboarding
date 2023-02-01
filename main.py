import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, ToTensor

from network import TurnNet
from dataset import CarDrivingDataset

# path for checkpoints
PATH = "checkpoint.pt"

transform = Compose([ToTensor(), Resize((66, 200))])
# see https://discuss.pytorch.org/t/dictionary-in-dataloader/40448 for proper use
loader = DataLoader(CarDrivingDataset(transform=transform), batch_size=8, shuffle=True)

model = TurnNet()
model.to("cpu")

# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# load from the checkpoint file
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']

model.train()

# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
# Run the training loop
for epoch in range(0, 5): # 5 epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(loader, 0):
        inputs = data["image"].float()
        targets = data["angle"].float()
        targets = torch.reshape(targets, (targets.shape[0], 1))
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 10 == 0:
            #print(f'Loss after mini-batch {i + 1:5d}: {current_loss / 10:.10f} for steering angle {outputs.tolist()}')
            current_loss = 0.0

    # save to checkpoint file
    print(f"Saving data from epoch {epoch + 1}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, PATH)

# Process is complete.
print('Training process has finished.')

torch.save(model.state_dict(), "save.pth")

pass