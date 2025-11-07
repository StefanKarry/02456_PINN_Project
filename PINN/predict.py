
import numpy as np
import torch
import matplotlib.pyplot as plt

model_script=
from lib/model_script import *


# Grid for prediction
N_x = 200  # antal punkter i x
N_t = 200  # antal punkter i t
x_star = np.linspace(-1, 1, N_x)
t_star = np.linspace(0, 5, N_t)  # vælg T_max efter dit problem
# Combine all the in a Matrix X (x,t) kombinationer
X_star = np.array([[x, t] for t in t_star for x in x_star])
X_star_tensor = torch.tensor(X_star, dtype=torch.float32)

model=SimpleNet()
#model=SimpleNet  find out what form the model is at/run at

#path_weights="model.pth" update this
model.load_state_dict(torch.load(path_weights))
model.eval()     #sæt i eval mode 

with torch.no_grad():  # we don't need gradients on the tensors because we are not training anymore
    u_pred = model(X_star_tensor)
    


    




