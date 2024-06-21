# import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
from monai.transforms import (
    AsDiscrete,
)

# !pip install monai
import monai
from PIL import Image
import torchvision.transforms.functional as F

torch.manual_seed(42)

device = torch.device("cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"Using device: {device}")

model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# Model pretrained on imagenet

# Larger transformation pipeline for imagenet
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((128, 128)),
    ]
)
try:
    datasets.SBDataset(
        root="./data/sbd",
        image_set="train",
        download=True,
        mode="segmentation",
    )
    datasets.SBDataset(
        root="./data/sbd",
        image_set="val",
        download=True,
        mode="segmentation",
    )
except Exception as e:
    print(e)

train_data = datasets.SBDataset(
    root="./data/sbd",
    image_set="train",
    transforms=lambda x, y: [transform(x), transform(y)],
    # download=True,
    mode="segmentation",
)

test_data = datasets.SBDataset(
    root="./data/sbd",
    image_set="val",
    transforms=lambda x, y: [transform(x), transform(y)],
    # download=True,
    mode="segmentation",
)


train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


X_test = F.to_pil_image(train_data[0][0]).save("X_test.png")
y_test = F.to_pil_image(train_data[0][1]).save("y_test.png")

learning_rate = 1e-4
epochs = 100
# Dice is a log loss function so negative values are expected
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = monai.losses.DiceLoss(sigmoid=True)

assert model(train_data[0][0].unsqueeze(0)).shape == (
    1,
    1,
    128,
    128,
), "Model output shape is correct"

model = model.to(device)
for epoch in tqdm(range(epochs)):
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        y = AsDiscrete(threshold=-0.9)(y).to(device)
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Epoch: {epoch+1}, Loss: {loss:.6f}, Progress: [{current}/{size}]")
