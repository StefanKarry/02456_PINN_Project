

import torch

# Domain limits
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_min, t_max = 0.0, 2.0  # adjust T as needed


N=[10000,1000,500]  # Number of points

N_f = N[0]  # interior PDE collocation points

# Random sampling
x_f = (x_max - x_min) * torch.rand(N_f, 1) + x_min
y_f = (y_max - y_min) * torch.rand(N_f, 1) + y_min
t_f = (t_max - t_min) * torch.rand(N_f, 1) + t_min

# Combine into X_f: shape (N_f, 3)
X_f = torch.cat([x_f, y_f, t_f], dim=1)


#--------------------------------------------------
N_ic = N[1]  # number of IC points

x_ic = (x_max - x_min) * torch.rand(N_ic, 1) + x_min
y_ic = (y_max - y_min) * torch.rand(N_ic, 1) + y_min
t_ic = torch.zeros(N_ic, 1)  # t=0 for IC

X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)


#--------------------------------------------------

N_bc_side = N[2]  # points per edge

# Left (x = x_min) and Right (x = x_max) edges
y_bc_lr = (y_max - y_min) * torch.rand(N_bc_side, 1) + y_min
t_bc_lr = (t_max - t_min) * torch.rand(N_bc_side, 1) + t_min

x_bc_left = x_min * torch.ones(N_bc_side, 1)
x_bc_right = x_max * torch.ones(N_bc_side, 1)

X_bc_left = torch.cat([x_bc_left, y_bc_lr, t_bc_lr], dim=1)
X_bc_right = torch.cat([x_bc_right, y_bc_lr, t_bc_lr], dim=1)

# Bottom (y = y_min) and Top (y = y_max) edges
x_bc_bt = (x_max - x_min) * torch.rand(N_bc_side, 1) + x_min
t_bc_bt = (t_max - t_min) * torch.rand(N_bc_side, 1) + t_min

y_bc_bottom = y_min * torch.ones(N_bc_side, 1)
y_bc_top = y_max * torch.ones(N_bc_side, 1)

X_bc_bottom = torch.cat([x_bc_bt, y_bc_bottom, t_bc_bt], dim=1)
X_bc_top = torch.cat([x_bc_bt, y_bc_top, t_bc_bt], dim=1)

# Combine all boundary points
X_bc = torch.cat([X_bc_left, X_bc_right, X_bc_bottom, X_bc_top], dim=0)
