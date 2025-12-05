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

from lib.loss.Loss_ninna import *
#from lib.dataset.dataset1 import *
#from Loss_2Dsquare import *
from lib.loss.Loss_sphere import *
from lib.model.PINNs import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
#Data set 
N=[10000,1000]
X_f, X_ic = WaveDataset(N, t_min=0.0, t_max=1.0)
X_f = X_f.to(device)
X_ic = X_ic.to(device)

#GPU memory is pretty good and we don't feel like training in batches for this PINN but otherwise we would use the DataLoader package:
#batch_size = 8
#trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model=PINNModel_Sphere().to(device)
criterion=loss_pinn_sphere 
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
summary(model,input_size=(3,))

# https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
# https://en.wikipedia.org/wiki/Limited-memory_BFGS
optimizer_finetune = optim.LBFGS(model.parameters(), lr = 1, max_iter = 5000, max_eval = 5000, history_size = 100, 
                                                     tolerance_grad = 1e-09, tolerance_change = 1.0 * np.finfo(float).eps,
                                                     line_search_fn = "strong_wolfe")




#folder for logs 
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = '/zhome/14/3/167963/02456_PINN_Project/PINN/lib/weights'
log_dir = os.path.join(current_dir, 'lib/logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_loss.txt')

# Training parameters
epochs = 30000
c = 1.4
counter = 0
loss_history = []

# Training loop
with open(log_file, 'w') as f:
    elapsed_time = 0.0

    for epoch in range(epochs):
        start_time = time()
        model.train()

        # Progressive data selection
        if (epoch) % (epochs // 10) == 0:
            counter += 1

        num_f = int(N[0] * (counter / 10))
        num_0 = int(N[1] * (counter / 10))

        x_f = X_f[:num_f, :]
        x_0 = X_ic[:num_0, :]

        # ADAM step
        optimizer.zero_grad()
        loss = criterion(model, x_f, x_0,c=c,beta_f=10,beta_ic=100,sigma=0.2,theta0=0.0,phi0=0.0,R=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss = loss.item()
        loss_history.append(epoch_loss)
        elapsed_time += time() - start_time

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, Time: {elapsed_time:.2f}s')

        # Log
        f.write(f"{epoch+1},{epoch_loss},{elapsed_time:.2f}\n")


    # Periodic L-BFGS fine-tune
    def closure():
        optimizer_finetune.zero_grad()
        loss_l = criterion(model, x_f, x_0,c=c,beta_f=10,beta_ic=100,sigma=0.2,theta0=0,phi0=0,R=1.0)
        loss_l.backward()
        return loss_l
    optimizer_finetune.step(closure)
    #log
    f.write(f"{epoch+1},{epoch_loss},{elapsed_time:.2f}\n")
    

#saving parameters (actuaylly overwriting the previous to save space)
save_path = os.path.join(model_dir, "PINNModel_Sphere.pth")

torch.save(model.state_dict(), save_path)
print(f'Model parameters saved to {save_path}') 
print('Training complete.')

