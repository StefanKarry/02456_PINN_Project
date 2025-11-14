import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time


# Generating data used for mappingg x to y
N = 10_000 #Number of samples
data = np.random.random(N).astype(np.float32).reshape(-1,1) #Input data (x)

train_size = int(0.75 * N)
X = data.reshape(-1, 1)  # Input data (x)
X_train = X[:train_size]
X_test = X[train_size:]


a, b, c = 2.5, -1.0, 0.5 #Coefficients of the polynomial function

def Y_poly(x, a, b, c):
    return a*x**3 + b*x**2 + c*x

def Y_sin(x):
    return np.sin(x)

def Y_abs(x):
    return np.abs(x)

def Y_arb(x):
    return np.exp(np.sin(x))

# Y_poly = a*X.flatten()**3 + b*X.flatten()**2 + c*X.flatten() #Output data (y) for polynomial function
# Y_sin = np.sin(X.flatten()) #Output data (y) for sine function
# Y_abs = np.abs(X.flatten()) #Output data (y) for absolute function
# Y_arb = np.exp(np.sin(X.flatten()))

#Convert to tensors for PyTorch
X_tensor = torch.tensor(X)


#  
# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

NN = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(NN.parameters(), lr = 0.001)

#Training loop
epochs = 50
batch_size = 5

for epoch in range(epochs):
    permutation = torch.randperm(train_size)
    epoch_loss = 0.0
    for i in range(0, train_size, batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_tensor[indices], torch.tensor(Y_poly(X_tensor[indices], a, b, c)).unsqueeze(1)
        optimizer.zero_grad()
        outputs = NN(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    if (epoch%10==9): print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/N:.6f}")


# Plotting the results
import matplotlib.pyplot as plt

NN.eval()
with torch.no_grad():
    predicted = NN(torch.tensor(X_test)).numpy()    
plt.figure(figsize=(10,6))
plt.scatter(X_test, Y_poly(X_test, a, b, c), label='True Data', color='blue', s=10
)
plt.scatter(X_test, predicted, label='NN Predictions', color='red', s=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Neural Network Function Approximation')
plt.legend()
plt.show()


