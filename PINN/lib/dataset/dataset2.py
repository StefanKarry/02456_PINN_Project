from turtle import width
import torch
import numpy as np
def create_training_data(x_min: float = None, x_max: float = None, t_min: float = None, t_max: float = None, N_f: float = None, N_b: float = None, N_0: float = None, sources = None, centers: list = None, sigma = None):
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
    # Interior Points (PDE)
    # Random sampling (x, t)
    # x in [-1, 1], t in [0, 1]
    x_f = (x_max - x_min) * torch.rand(N_f, 1) + x_min
    t_f = (t_max - t_min) * torch.rand(N_f, 1) + t_min

    # Combine them into one tensor requiring gradients
    X_f = torch.cat([x_f, t_f], dim=1)
    
    #Sort based on time for causal training
    sorted_indices = torch.argsort(X_f[:, 1])
    X_f = X_f[sorted_indices].requires_grad_(True)


    # Boundary Points (Neumann)
    
    # Left Edge: x = -1
    x_b_left = torch.ones(N_b, 1) * x_min
    t_b_left = (t_max - t_min) * torch.rand(N_b, 1) + t_min
    X_b_left = torch.cat([x_b_left, t_b_left], dim=1)

    # Sort based on time for causal training
    sorted_indices_left = torch.argsort(X_b_left[:, 1])
    X_b_left = X_b_left[sorted_indices_left].requires_grad_(True)

    # Right Edge: x = 1
    x_b_right = torch.ones(N_b, 1) * x_max
    t_b_right = (t_max - t_min) * torch.rand(N_b, 1) + t_min
    X_b_right = torch.cat([x_b_right, t_b_right], dim=1)

    # Sort based on time for causal training
    sorted_indices_right = torch.argsort(X_b_right[:, 1])
    X_b_right = X_b_right[sorted_indices_right].requires_grad_(True)

    # --- 3. Initial Condition Points ---
    # t = 0, x is random
    if (len(centers) < 2) or (sources is None or 1): #By default use single source
        x_0 = (x_max - x_min) * torch.rand(N_0, 1) + x_min
        t_0 = torch.zeros(N_0, 1)
        X_0 = torch.cat([x_0, t_0], dim=1)

        # Sort based on time for causal training
        sorted_indices_0 = torch.argsort(X_0[:, 1])
        X_0 = X_0[sorted_indices_0].requires_grad_(True)

    else:            
        center_s1 = sources[0]
        center_s2 = sources[1]   # Location of the second source
        sigma = 0.2       # Area around the source to sample densely
        
        n_source = N_0 // 3  # 1/3rd of points for Source 1
        n_source2 = N_0 // 3 # 1/3rd of points for Source 2
        n_bg = N_0 - n_source - n_source2 # Remaining for background
        
        # Sample densely around Source 1
        x_0_s1 = (2 * sigma) * torch.rand(n_source, 1) + (center_s1 - sigma)
        
        # Sample densely around Source 2
        x_0_s2 = (2 * sigma) * torch.rand(n_source2, 1) + (center_s2 - sigma)
        
        # Sample the rest of the domain (Background) to ensure it stays flat
        x_0_bg = (x_max - x_min) * torch.rand(n_bg, 1) + x_min
        
        # All the sampled x_0 points
        x_0 = torch.cat([x_0_s1, x_0_s2, x_0_bg], dim=0)
        t_0 = torch.zeros(N_0, 1)
        X_0 = torch.cat([x_0, t_0], dim=1)

        # SHUFFLE X_0
        perm = torch.randperm(X_0.size(0))
        X_0 = X_0[perm].requires_grad_(True)

    return X_f, X_b_left, X_b_right, X_0