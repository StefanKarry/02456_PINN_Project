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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINN_Model_1D().to(device)

# Load trained model parameters and loss history
current_dir = os.path.dirname(os.path.abspath(__file__)) 
model_dir = os.path.join(current_dir, 'lib/weights')

# If moving between 1 source and 2 source models, change the filename accordingly and go to exact_1D_grid.py and change SOURCES to 1 or 2.

load_path = os.path.join(model_dir, f'{model.__class__.__name__}_1D_1source.pth') #Include _good for the good model
model.load_state_dict(torch.load(load_path, map_location=device))

loss_history = np.load(os.path.join(model_dir, f'{model.__class__.__name__}_1D_loss_history_1source.npy')) # include _good for the good model

#### Domain and Wave Params ####
DOMAIN_START = -1.0
DOMAIN_END = 1.0
T_MAX_PLOT = 2.0
WAVE_SPEED = 1.4
N_TERMS = 50  # Number of Fourier terms to use

# Evaluating performance
x_eval = np.linspace(DOMAIN_START, DOMAIN_END, 200)
t_eval = np.linspace(0, T_MAX_PLOT, 200) # Plot up to 1 second
X_grid, T_grid = np.meshgrid(x_eval, t_eval)

# Computing exact solution for comparison
print("Computing Exact Solution...")
U_exact = get_solution_grid(X_grid, T_grid)

# Prediction (should probably be moved to another script and weights saved/loaded)
print("Predicting with PINN model...")
model.eval()
x_flat = X_grid.flatten()[:, None]
t_flat = T_grid.flatten()[:, None]

x_tensor = torch.tensor(x_flat, dtype=torch.float32)
t_tensor = torch.tensor(t_flat, dtype=torch.float32)

with torch.no_grad():
    u_pred_tensor = model(x_tensor.to(device), t_tensor.to(device))

U_pinn = u_pred_tensor.cpu().numpy().reshape(X_grid.shape)

# Absolute error
Error_grid = np.abs(U_exact - U_pinn)
# Error_grid = U_exact - U_pinn

# Result visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vmin = np.min(U_exact)
vmax = np.max(U_exact)

# Exact solution
# im1 = axes[0].pcolormesh(X_grid, T_grid, U_exact, cmap='turbo', vmin=vmin, vmax=vmax, shading='auto')
# axes[0].set_title("Exact Solution (Fourier)")
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("t")
# plt.colorbar(im1, ax=axes[0])

# PINN prediction
im2 = axes[0].pcolormesh(X_grid, T_grid, U_pinn, cmap='turbo', vmin=vmin, vmax=vmax, shading='auto')
axes[0].set_title("PINN Prediction")

if T_MAX_PLOT > 1.0:
    axes[0].axhline(y=T_MAX_PLOT - (T_MAX_PLOT - 1.0), color='white', linestyle='--', label='Out of\nTraining\nDomain')
    axes[0].legend(loc = 'lower left')

axes[0].set_xlabel("Position ($x$)")
axes[0].set_ylabel("Time ($t$)")
plt.colorbar(im2, ax=axes[0], label='Displacement $u(x,t)$')

# Error plot
im3 = axes[1].pcolormesh(X_grid, T_grid, Error_grid, cmap='inferno', shading='auto')

if T_MAX_PLOT > 1.0:
    axes[1].axhline(y=T_MAX_PLOT - (T_MAX_PLOT - 1.0), color='white', linestyle='--', label='Out of\nTraining\nDomain')
    axes[1].legend(loc = 'lower left')

axes[1].set_title("Absolute Error |Exact - PINN|")
axes[1].set_xlabel("Position ($x$)")
axes[1].set_ylabel("Time ($t$)")
plt.colorbar(im3, ax=axes[1], label='Absolute Error')

# im4 = axes[3].plot(np.log(loss_history))
# axes[3].set_title("Training Loss History")
# axes[3].set_xlabel("Epoch")
# axes[3].set_ylabel("log(Loss)")
# axes[3].grid(True)

plt.tight_layout()
plt.savefig("PINN_1D_Wave_Equation_Results.png", bbox_inches='tight', dpi=300)

fig, axes = plt.subplots(1, 1, figsize=(12, 5))
im4 = axes.plot(np.log(loss_history))
axes.set_title("Training Loss History")
axes.set_xlabel("Epoch")
axes.set_ylabel("log(Loss)")
axes.grid(True)
plt.tight_layout()
plt.savefig("PINN_1D_Wave_Equation_Loss_History.png", bbox_inches='tight', dpi=300)
