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
import matplotlib.pyplot as plt

# Our scripts
from lib.loss.Loss_1D import compute_grad, waveLoss_1D
from lib.dataset.dataset_1D import create_training_data
from lib.model.PINNs import PINN_Model_1D
from lib.dataset.exact_1D_grid import *

#### Domain and Wave Params ####
DOMAIN_START = -1.0
DOMAIN_END = 1.0
T_MAX_PLOT = 1.0
WAVE_SPEED = 1.4
N_TERMS = 50  # Number of Fourier terms to use
SOURCES = 2

# Loss function weights
w_F = 10.0
w_IC = 100.0
w_B = 100.0

N_f = 1000 * 10 # Number of collocation points
N_b = 200 * 10 # Number of boundary points (this is doubled in practice, since we have both left and right sides)
N_0 = 200 * 10 # Number of initial condition points

# Data generation. Sliced during training for causal training
X_f, X_b_left, X_b_right, X_0 = create_training_data(x_min=DOMAIN_START, x_max=DOMAIN_END, t_min=0.0, t_max=T_MAX_PLOT, N_f=N_f, N_b=N_b, N_0=N_0, sources = 2, centers = [0.5, -0.3], sigma = 0.2)

# Initial condition target values   
u_0_target = torch.tensor(initial_displacement(X_0[:, 0:1].detach().numpy(), sources = SOURCES), dtype=torch.float32)


#### Training setup ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
epochs = 30_000

model = PINN_Model_1D().to(device)
criterion = waveLoss_1D # Use the custom loss function 
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
# https://en.wikipedia.org/wiki/Limited-memory_BFGS
optimizer_finetune = optim.LBFGS(model.parameters(), lr = 1, max_iter = 5000, max_eval = 5000, history_size = 100, 
                                                     tolerance_grad = 1e-09, tolerance_change = 1.0 * np.finfo(float).eps,
                                                     line_search_fn = "strong_wolfe")

loss_history = []

#### Training loop ####
counter = 0

elapsed_time = 0.0
for epoch in range(epochs):
    start_time = time()
    model.train()

    # Dynamic data selection for causal training, gradually increasing the number of training points/domain coverage. 
    if (epoch) % ((epochs) // 10) == 0:
        print(f"Epoch {epoch}/{epochs}")
        counter += 1

        num_f = int(N_f * (counter / 10)) 
        num_b = int(N_b * (counter / 10))
        num_0 = int(N_0 * (counter / 10))

        x_f, t_f = X_f[0:num_f, 0:1], X_f[0:num_f, 1:2]
        x_b_left, t_b = X_b_left[0:num_b, 0:1], X_b_left[0:num_b, 1:2]
        x_b_right, t_b = X_b_right[0:num_b, 0:1], X_b_right[0:num_b, 1:2]
        x_0, t_0 = X_0[0:num_0, 0:1], X_0[0:num_0, 1:2]

        u_0_target = torch.tensor(initial_displacement(x_0.detach().numpy(), sources = SOURCES), dtype=torch.float32)
        print(f"Training data points - Collocation: {x_f.shape[0]}, Boundary: {x_b_left.shape[0] + x_b_right.shape[0]}, Initial: {x_0.shape[0]}")

    # Uses ADAM optimizer for initial training
    optimizer.zero_grad()

    loss = criterion(model, x_f.to(device),
                     t_f.to(device),x_b_left.to(device),
                     x_b_right.to(device), t_b.to(device), x_0.to(device),
                     t_0.to(device), u_0_target.to(device),
                     c=WAVE_SPEED, beta_f=w_F, beta_ic=w_IC, beta_b=w_B)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #Handles exploding gradients
    optimizer.step()

    epoch_loss = loss.item()
    loss_history.append(epoch_loss)
    elapsed_time += time() - start_time
    if (epoch+1) % 300 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s")

# Uses L-BFGS optimizer for fine-tuning (Only does once after ADAM since it loops internally)
optimizer = optimizer_finetune
print("Switched to L-BFGS optimizer for fine-tuning on full dataset.")

# By recommendation from Gemini, add a resampling step before L-BFGS optimization, such that the optimizer only focus on high error areas
# This ensured it doesn't waste resources on areas where the model is already performing well (background)

print('Resampling training data for L-BFGS optimization...')
# 1. Generate candidate points
x_cand = (DOMAIN_END - DOMAIN_START) * torch.rand(50000, 1) + DOMAIN_START
t_cand = (T_MAX_PLOT - 0) * torch.rand(50000, 1) + 0
X_cand = torch.cat([x_cand, t_cand], dim=1).to(device).requires_grad_(True)

x_in = X_cand[:, 0:1]
t_in = X_cand[:, 1:2]
# 2. Evaluate Residual (No training, just checking)
u_pred = model(x_in, t_in)
u_t = compute_grad(u_pred, t_in)
u_x = compute_grad(u_pred, x_in)
u_tt = compute_grad(u_t, t_in)
u_xx = compute_grad(u_x, x_in)
res = torch.abs(u_tt - WAVE_SPEED**2 * u_xx).detach().flatten()

# Selecting points with the highest residuals
_, indices = torch.topk(res.squeeze(), k=10000) # Select top 10,000 points with highest residuals
X_finetune = X_cand[indices]

x_f = X_finetune[:, 0:1].detach().requires_grad_(True)
t_f = X_finetune[:, 1:2].detach().requires_grad_(True)

print(f"Finetune collocation points selected: {x_f.shape[0]}")

# Requirement for L-BFGS optimizer
if isinstance(optimizer, optim.LBFGS):
    def closure():
        optimizer.zero_grad()
        loss = criterion(model, x_f.to(device),
                         t_f.to(device), x_b_left.to(device),
                         x_b_right.to(device),t_b.to(device),
                         x_0.to(device), t_0.to(device), u_0_target.to(device),
                         c=WAVE_SPEED, beta_f=w_F, beta_ic=w_IC, beta_b=w_B)   
        loss.backward()
        return loss

    optimizer.step(closure)
    epoch_loss = closure().item()
    loss_history.append(epoch_loss)
    elapsed_time += time() - start_time

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.9f}, Time: {elapsed_time:.2f}s")


# Saving the weights
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'lib/weights')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
save_path = os.path.join(model_dir, f'{model.__class__.__name__}_1D_test.pth')

torch.save(model.state_dict(), save_path)
print(f'Model parameters saved to {save_path}') # These can be used for inference later on, since training took ~18 minutes

#Saving training history
history_path = os.path.join(model_dir, f'{model.__class__.__name__}_1D_loss_history_test.npy')
np.save(history_path, np.array(loss_history))
print(f'Training loss history saved to {history_path}')

