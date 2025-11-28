import torch
import numpy as np

# Domain bounds
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0 # Let's assume max time is 1 second

# Number of points
N_f = 4000 # Interior physics points
N_b = 250  # Boundary points (per side)
N_0 = 250  # Initial points

def create_training_data(x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max, N_f=N_f, N_b=N_b, N_0=N_0):
    """
    Create training data for the PINN model solving the 1D wave equation.
    This includes interior points for the PDE residual, boundary points for Neumann BCs,
    and initial condition points.
    :Parameters:
    - x_min, x_max: Spatial domain bounds
    - t_min, t_max: Temporal domain bounds
    - N_f: Number of interior points
    - N_b: Number of boundary points (per side)
    - N_0: Number of initial condition points
    :Returns:
    - X_f: Interior points (x, t) for PDE residual
    - X_b_left: Boundary points at left edge (x=-1, t) for Neumann BC
    - X_b_right: Boundary points at right edge (x=1, t) for Neumann BC
    - X_0: Initial condition points (x, t=0)
    """
    # --- 1. Interior Points (PDE) ---
    # Random sampling (x, t)
    # x in [-1, 1], t in [0, 1]
    x_f = (x_max - x_min) * torch.rand(N_f, 1) + x_min
    t_f = (t_max - t_min) * torch.rand(N_f, 1) + t_min
    # Combine them into one tensor requiring gradients
    X_f = torch.cat([x_f, t_f], dim=1).requires_grad_(True)

    # --- 2. Boundary Points (Neumann) ---
    # Left Edge: x = -1, t is random
    x_b_left = torch.ones(N_b, 1) * x_min
    t_b_left = (t_max - t_min) * torch.rand(N_b, 1) + t_min
    X_b_left = torch.cat([x_b_left, t_b_left], dim=1).requires_grad_(True)

    # Right Edge: x = 1, t is random
    x_b_right = torch.ones(N_b, 1) * x_max
    t_b_right = (t_max - t_min) * torch.rand(N_b, 1) + t_min
    X_b_right = torch.cat([x_b_right, t_b_right], dim=1).requires_grad_(True)

    X_b = torch.cat([X_b_left, X_b_right], dim=0).requires_grad_(True)


    # --- 3. Initial Condition Points ---
    # t = 0, x is random
    x_0 = (x_max - x_min) * torch.rand(N_0, 1) + x_min
    t_0 = torch.zeros(N_0, 1)
    X_0 = torch.cat([x_0, t_0], dim=1).requires_grad_(True)

    return X_f, X_b, X_0