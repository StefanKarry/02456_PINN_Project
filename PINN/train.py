import os
import numpy as np
import glob
import PIL.Image as Image
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

from PINN.lib.loss.Loss_ninna import *
from lib.dataset.dataset1 import *
from PINN.lib.loss.Loss_2Dsquare import *
from Loss_sphere import *
from lib.model.PINNs import *

#### Data set ####
batch_size = 8
trainset = WaveDataset(num_samples=1000, train=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

#### Training setup ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50

#model = PINNModel_Plane().to(device)
#criterion = loss_pinn_2D() # Use the custom loss function 
model=PINNModel_Sphere().to(device)
criterion=loss_pinn_sphere() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)
summary(model, )(input_size=(2,))

# -------------------
#PARAMS 2D
# -------------------
c = 1.4 # wave speed (used in the loss function)
#x0, y0 = 1.0, 1.0 (initial position of wave, default in Loss is 0.0, 0.0)
#sigma=2 (initial condition wave width, default in Loss is 1)
#beta_f=1.0 and beta_ic=1.0: weights of each of the loss terms (hyperparameters to be tuned - i.e. NOT learned during training by default)

# Domain params
#x_min, x_max = -1.0, 1.0 
#y_min, y_max = -1.0, 1.0
#t_min, t_max = 0.0, 2.0  # adjust T as needed (default values are 0.0,2.0)
#N=[10000,1000,500] (default is [10000,1000,500])  # Number of points in the grid for each coordinate


# -------------------
#PARAMS spherical
# -------------------
c = 1.4 # wave speed (used in the loss function)
#theta0,phi0=5.0,2.0 (initial position of wave, default in Loss is 1.0, 1.0)
#sigma=5 (initial condition wave width, default in Loss is 0.2)
#R=3.0 (radius of sphere. Default is 1.0)
#beta_f=1.0 and beta_ic=1.0: weights of each of the loss terms

# Domain params
#t_min, t_max = 0.0, 2.0  (default values are 0.0,2.0) # adjust T as needed
#N=[10000,1000]  (default is [10000,1000]) # Number of points in the grid for each coordinate




#folder for logs 
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'lib/weights')
log_dir = os.path.join(current_dir, 'lib/logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_loss.txt')



#### Training loop ####
with open(log_file, 'w') as f:
    elapsed_time = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time()
        for i, (inputs, target) in enumerate(trainloader):
            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, c)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        elapsed_time += time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(trainloader):.4f}, Time: {elapsed_time:.2f}s")
        
        # write to our training loss log txt file
        f.write(f"{epoch+1},{epoch_loss},{elapsed_time:.2f}\n")
        f.flush()  # make sure that we're writing to disk every epoch

#### Save model parameters ####
save_path = os.path.join(model_dir, f'{model.__class__.__name__}.pth')

torch.save(model.state_dict(), save_path)
print(f'Model parameters saved to {save_path}') # These can be used for inference.
print('Training complete.')

