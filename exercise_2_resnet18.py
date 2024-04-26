# Learning point:
# Working with RGB images and pretrained models
# Using a pretrained model for transfer learning
# Task:
# Train a ResNet18 model on the CIFAR10 dataset
# Learn how to add data augmentation to the transformation pipeline

# Note: MLP will struggle with CIFAR10 dataset due to its complexity
# Note: CIFAR10 dataset has 3 channels (RGB) and 10 classes
# Note: ResNet18 is a popular model for image classification tasks
# Note: CIFAR10 dataset has 60,000 32x32 color images in 10 classes, with 6,000 images per class
# Note: The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models

torch.manual_seed(42)

device = torch.device("cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"Using device: {device}")
# Model pretrained on imagenet

resnet18 = models.resnet18(pretrained=True).to(device)
fc = nn.Linear(1000, 10)
model = nn.Sequential(resnet18, fc)
# Larger transformation pipeline for imagenet
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_data = datasets.CIFAR10(
    root="./data/CIFAR10", train=True, transform=transform, download=True
)
test_data = datasets.CIFAR10(
    root="./data/CIFAR10", train=False, transform=transform, download=True
)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=True,
)


learning_rate = 1e-3
epochs = 100
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model = model.to(device)

for epoch in tqdm(range(epochs)):
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        # loss = loss_fn(y, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Epoch: {epoch+1}, Loss: {loss:.6f}, Progress: [{current}/{size}]")


# Test the model cross validation


model.eval()
y_pred = []
y_true = []

# Disable gradient computation for evaluation to save memory and computations
with torch.no_grad():
    all_preds = []
    all_labels = []

    for X, y in test_dataloader:
        X = X.device()  # No device assignment since it defaults to CPU in your script
        preds = model(X).to(device).cpu()
        all_preds.extend(preds.argmax(1).numpy())  # Get the predicted classes
        all_labels.extend(y.numpy())

# Convert list to NumPy arrays for Scikit-Learn
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Classification report
report = classification_report(
    all_labels, all_preds, target_names=[str(i) for i in range(10)]
)


print("Classification Report:\n", report)
