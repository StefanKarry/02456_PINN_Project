
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from lib.model.PINNs import *


# Grid for prediction
N_x = 200  # antal punkter i x
N_t = 200  # antal punkter i t
x_star = np.linspace(-1, 1, N_x)
t_star = np.linspace(0, 5, N_t)  # vælg T_max efter dit problem
# Combine all the in a Matrix X (x,t) kombinationer
X_star = np.array([[x, t] for t in t_star for x in x_star])
X_star_tensor = torch.tensor(X_star, dtype=torch.float32)

model=PINNModel_Plane()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
working_dir = os.path.dirname(os.path.abspath(__file__))
path_weights = os.path.join(working_dir, 'lib/weights/model.pth') # Works dynamically for any of our systems

model.load_state_dict(torch.load(path_weights))
model.eval()     #sæt i eval mode 

with torch.no_grad():  # we don't need gradients on the tensors because we are not training anymore
    X_star_tensor = X_star_tensor.to(device)
    u_pred = model(X_star_tensor)
    


    




