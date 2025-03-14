# -*- coding: utf-8 -*-
"""digit_recognization.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZMhQLsKBUrCfZ3i2KUw78GJ7SEubj4wQ
"""

!pip install torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self): # Changed _init_ to __init__
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Flatten the 28x28 images into a vector
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Output 10 classes (digits 0-9)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from google.colab import drive
drive.mount('/content/drive')

epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Finished Training")

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get one batch of test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Show the first image in the batch
# imshow(images[1])

# Print the predicted label
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    for i in range(len(predicted)):
      imshow(images[i])
      print(f'Predicted Label: {predicted[i]}')
      print(f'True Label: {predicted[i]}')