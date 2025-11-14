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

from lib.model.model1 import *
from lib.Loss_ninna import *
from lib.dataset.dataset1 import *
from lib/Loss_2Dsquare import *

#### Data set ####
batch_size = 8
trainset = WaveDataset(num_samples=1000, train=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

#### Training setup ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50

model = PINNModel().to(device)
c = 1.4 # wave speed (used in the loss function)
x0, y0 = 0.0, 0.0
criterion = loss_pinn_2D() # Use the custom loss function 
optimizer = optim.Adam(model.parameters(), lr=1e-4)

summary(model, )(input_size=(2,))

#### Training loop ####
elapsed_time = 0.0
for epoch in range(epochs):
    epoch_loss = 0.0
    start_time = time()
    for i, (inputs, target) in enumerate(trainloader):
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target, c)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    elapsed_time += time() - start_time
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(trainloader):.4f}, Time: {elapsed_time:.2f}s")

#### Save model parameters ####
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'lib/weights')
save_path = os.path.join(model_dir, f'{model.__class__.__name__}.pth')

torch.save(model.state_dict(), save_path)
print(f'Model parameters saved to {save_path}') # These can be used for inference.
print('Training complete.')

