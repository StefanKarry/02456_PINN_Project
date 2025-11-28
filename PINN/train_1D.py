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
from lib.loss.Loss_ninna import compute_grad, waveLoss_2D
from lib.dataset.dataset2 import N_b, create_training_data
from lib.model.PINNs import PINN_Model_2D
from lib.dataset.exact_1D_grid import *

#### Data set ####
batch_size = 8

# We try implementing Causal Training which trains the model in a time marching manner.
# This means that it i.e. learns to predict the wave propagation from t=0 to t=0.1 first, then from t=0.1 to t=0.2, etc.
# For this, we need to create training data for the entire domain first, then we can slice it up in time segments during training.




#### Domain and Wave Params ####
DOMAIN_START = -1.0
DOMAIN_END = 1.0
T_MAX_PLOT = 1.0
WAVE_SPEED = 1.2
N_TERMS = 100  # Number of Fourier terms to use

X_f, X_b, X_0 = create_training_data(x_min=DOMAIN_START, x_max=DOMAIN_END, t_min=0.0, t_max=T_MAX_PLOT, N_f=5000*10, N_b=500*10, N_0=500*10)

#Creating data slices for training. Covers different areas of the domain.

x_f, t_f = X_f[:, 0:1], X_f[:, 1:2]
x_b, t_b = X_b[:, 0:1], X_b[:, 1:2]
x_0, t_0 = X_0[:, 0:1], X_0[:, 1:2]



u_0_target = torch.tensor(initial_displacement(X_0[:, 0:1].detach().numpy()), dtype=torch.float32)

quit()

#### Training setup ####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
epochs = 10_000

model = PINN_Model_2D().to(device)
criterion = waveLoss_2D # Use the custom loss function 
#criterion=loss_pinn_sphere() #Use custom loss function for spherical domain 
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

loss_history = []

#### Training loop ####

c = 0

elapsed_time = 0.0
for i in range(epochs):
    start_time = time()
    model.train()

    optimizer.zero_grad()

    loss = criterion(model, x_f.to(device), t_f.to(device), x_b.to(device), t_b.to(device), x_0.to(device), t_0.to(device), u_0_target.to(device), c=WAVE_SPEED, beta_f=1.2, beta_ic=1.3, beta_b=1.3)
    loss.backward()
    optimizer.step()

    epoch_loss = loss.item()
    loss_history.append(epoch_loss)
    elapsed_time += time() - start_time
    if (i+1) % 100 == 0 or i == 0:
        print(f"Epoch [{i+1}/{epochs}], Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s")


# Evaluating performance
x_eval = np.linspace(DOMAIN_START, DOMAIN_END, 200)
t_eval = np.linspace(0, 1.0, 200) # Plot up to 1 second
X_grid, T_grid = np.meshgrid(x_eval, t_eval)

# Computing exact solution for comparison
print("Computing Exact Solution...")
U_exact = get_solution_grid(X_grid, T_grid)

#Predicting with the trained model
print("Predicting with PINN model...")
model.eval()
x_flat = X_grid.flatten()[:, None]
t_flat = T_grid.flatten()[:, None]

x_tensor = torch.tensor(x_flat, dtype=torch.float32)
t_tensor = torch.tensor(t_flat, dtype=torch.float32)

test_loss = []
with torch.no_grad(): # specific context that turns off gradient tracking to save memory
    u_pred_tensor = model(x_tensor.to(device), t_tensor.to(device))
# Reshape back to the 2D Grid shape for plotting
U_pinn = u_pred_tensor.cpu().numpy().reshape(X_grid.shape)

# 4. Calculate Absolute Error
Error_grid = np.abs(U_exact - U_pinn)

# --- PLOTTING ---
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Shared settings for color scaling (so the plots match visually)
vmin = np.min(U_exact)
vmax = np.max(U_exact)

# Plot 1: Exact
im1 = axes[0].pcolormesh(X_grid, T_grid, U_exact, cmap='seismic', vmin=vmin, vmax=vmax, shading='auto')
axes[0].set_title("Exact Solution (Fourier)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
plt.colorbar(im1, ax=axes[0])

# Plot 2: PINN
im2 = axes[1].pcolormesh(X_grid, T_grid, U_pinn, cmap='seismic', vmin=vmin, vmax=vmax, shading='auto')
axes[1].set_title("PINN Prediction")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
plt.colorbar(im2, ax=axes[1])

# Plot 3: Error
# We use a different colormap (Reds) to highlight error hotspots
im3 = axes[2].pcolormesh(X_grid, T_grid, Error_grid, cmap='inferno', shading='auto')
axes[2].set_title("Absolute Error |Exact - PINN|")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
plt.colorbar(im3, ax=axes[2])

im4 = axes[3].plot(loss_history)
axes[3].set_title("Training Loss History")
axes[3].set_xlabel("Epoch")
axes[3].set_ylabel("Loss")
axes[3].grid(True)

plt.tight_layout()
plt.show()

# #### Save model parameters ####
# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_dir = os.path.join(current_dir, 'lib/weights')
# save_path = os.path.join(model_dir, f'{model.__class__.__name__}.pth')

# torch.save(model.state_dict(), save_path)
# print(f'Model parameters saved to {save_path}') # These can be used for inference.
# print('Training complete.')

