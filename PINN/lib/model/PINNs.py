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

class PINNModel_Plane(nn.Module):
    '''
    A Physics Informed Neural Network model for simulating wave propagation on a 2D plane.
    The model takes as input the spatial coordinates (x, y) and time t, and outputs the wave displacement u.
    It practically solves an IVP, such that given some initial conditions (x, y) at t=0, it can predict the wave propagation over time.
    
    :Parameters:
    - x: x-coordinate in 2D space
    - y: y-coordinate in 2D space
    - t: Time variable

    :Output:
    - u: Wave displacement at the given (x, y, t)
    '''

    def __init__(self):
        super(PINNModel_Plane, self).__init__()
        self.layer1 = nn.Linear(3, 50) # input layer containing x, y, t
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 1) # output layer containing u(x, y, t)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))  
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class PINNModel_Sphere(nn.Module):
    '''
    A Physics Informed Neural Network model for simulating wave propagation on a spherical surface.
    The model takes as input the spherical coordinates (theta, phi) and time t, and outputs the wave displacement u.
    It practically solves an IVP, such that given some initial conditions (theta, phi) at t=0, it can predict the wave propagation over time.
    
    :Parameters:
    - theta: Polar angle in spherical coordinates (0 <= theta <= pi)
    - phi: Azimuthal angle in spherical coordinates (0 <= phi < 2*pi)
    - t: Time variable

    :Output:
    - u: Wave displacement at the given (theta, phi, t)
    '''
    def __init__(self):
        super(PINNModel_Sphere, self).__init__()
        self.layer1 = nn.Linear(3, 100) # input layer containing theta, phi, t
        self.layer2 = nn.Linear(100, 100)   
        self.layer3 = nn.Linear(100, 1) # output layer containing u(theta, phi, t)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class PINN_Model_2D(nn.Module):
    def __init__(self):
        super(PINN_Model_2D, self).__init__()
        self.layer1 = nn.Linear(2, 32) # input layer containing x, t
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32) 
        self.layer4 = nn.Linear(32, 1) # output layer containing u(x, t)
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        x = torch.tanh(self.layer1(inputs))  # Using tanh activation since it's smooth and works well for PINNs (maybe a insert reference here)
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = self.layer4(x)
        return x


