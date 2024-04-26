#%% Learning point: understand basic PyTorch concepts and how to train a simple MLP model

"""
Task:  fix this buggy code in the code
"""
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

torch.manual_seed(42)

#%%  

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # breakpoint()
        # Flatten the image for the input layer
        x = self.flatten(x)
        # Apply the linear layers of MLP with ReLU activation
        logits = self.linear_relu_stack(x)
        # Apply the softmax function to get probabilities
        probabilities = self.softmax(logits)
        return probabilities

#%%  
# Load MNIST dataset
# Import to rescale the image to [-1, 1] to match activation functions
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Load the MNIST dataset
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
#%%  
# Prepare the dataloaders, shuffle the data, and set the batch size
# Batches are used to update the model weights because we can't pass the entire dataset at once
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#%%  
# Initialize the model
model = NeuralNetwork()
print(model)
#%%  
# MNIST dataset
test_tensor = train_data[0]
print("Image shape:", test_tensor[0].shape, "Class:", test_tensor[1])
#%%  
print("Test model forward pass")
assert model(test_tensor[0]).shape == (1, 10), "Model output shape is incorrect"
#%%  
learning_rate = 1e-3
epochs = 25
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% Training 
for epoch in tqdm(range(epochs)):
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        # Clear old gradients
        optimizer.zero_grad()
        # Compute derivatives
        loss.backward()
        # Update the weights of the model using the optimizer
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Epoch: {epoch+1}, Loss: {loss:.6f}, Progress: [{current}/{size}]")

#%%  
# Test the model
model.eval()
y_pred = []
y_true = []

# Disable gradient computation for evaluation to save memory and computations
with torch.no_grad():
    all_preds = []
    all_labels = []

    for X, y in test_dataloader:
        # breakpoint()
        preds = model(X)
        all_preds.extend(preds.argmax(1).numpy())  # Get the predicted classes
        all_labels.extend(y.numpy())
#%%  
# Convert list to NumPy arrays for Scikit-Learn
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
#%%  
# Classification report
report = classification_report(
    all_labels, all_preds, target_names=[str(i) for i in range(10)]
)
print("Classification Report:\n", report)
