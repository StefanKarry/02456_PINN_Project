

import torch

# Domain limits
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0  # adjust T as needed

# Number of points
N_f = 10000  # interior PDE collocation points

# Random sampling
x_f = (x_max - x_min) * torch.rand(N_f, 1) + x_min
y_f = (y_max - y_min) * torch.rand(N_f, 1) + y_min
t_f = (t_max - t_min) * torch.rand(N_f, 1) + t_min

# Combine into X_f: shape (N_f, 3)
X_f = torch.cat([x_f, y_f, t_f], dim=1)


#--------------------------------------------------
N_ic = 1000  # number of IC points

x_ic = (x_max - x_min) * torch.rand(N_ic, 1) + x_min
y_ic = (y_max - y_min) * torch.rand(N_ic, 1) + y_min
t_ic = torch.zeros(N_ic, 1)  # t=0 for IC

X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
