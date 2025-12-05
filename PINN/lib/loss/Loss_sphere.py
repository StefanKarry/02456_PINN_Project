import torch
from torch.utils.data import Dataset
import numpy as np

# -------------------
# Initial Gaussian on the sphere (for t=0 condition)
# -------------------
def psi_sphere(theta, phi, theta0=1.0, phi0=1.0, sigma=0.2,R=1):
    #return torch.exp(-((theta-theta0)**2 + (phi-phi0)**2)/sigma**2)
    #return ((theta-theta0)**2 - (phi-phi0)**2)/sigma**2
    device = theta.device  # ensures everything is on the same device
    theta0 = torch.tensor(theta0, dtype=theta.dtype, device=device)
    phi0   = torch.tensor(phi0,   dtype=phi.dtype,   device=device)
    
    x0 = R * torch.sin(theta0) * torch.cos(phi0)
    y0 = R * torch.sin(theta0) * torch.sin(phi0)
    x  = R * torch.sin(theta) * torch.cos(phi)
    y  = R * torch.sin(theta) * torch.sin(phi)
    
    return (x - x0)**2 - (y - y0)**2

# -----------------------------
# Wave dataset for 3D spherical domain
# -----------------------------
def WaveDataset(N=[10000,1000], t_min=0.0, t_max=2.0):
    N_f, N_ic = N
    theta_f = torch.rand(N_f,1) * np.pi
    phi_f   = torch.rand(N_f,1) * 2*np.pi
    t_f     = t_min + (t_max - t_min) * torch.rand(N_f,1)
    X_f = torch.cat([theta_f, phi_f, t_f], dim=1)
    theta_ic = torch.rand(N_ic,1) * np.pi
    phi_ic   = torch.rand(N_ic,1) * 2*np.pi
    t_ic     = torch.zeros(N_ic,1)
    X_ic = torch.cat([theta_ic, phi_ic, t_ic], dim=1)
    return X_f, X_ic   


# -------------------
# Spherical PINN Loss
# -------------------
def loss_pinn_sphere(model_hat, X_f, X_ic,c, beta_f=1.2, beta_ic=1.2, sigma=0.2,theta0=1.0, phi0=1.0, R=1.0):
    # ------------------
    # PDE Loss (interior points)
    # ------------------
    X_f.requires_grad_(True)  #activate so we can compute the gradients 
    theta_f = X_f[:,0:1]
    phi_f   = X_f[:,1:2]
    u_f = model_hat(X_f)

    grads_f = torch.autograd.grad(u_f, X_f,
                                  grad_outputs=torch.ones_like(u_f),
                                  retain_graph=True, create_graph=True)[0]
    u_t = grads_f[:,2:3]
    u_tt = torch.autograd.grad(u_t, X_f,
                               grad_outputs=torch.ones_like(u_t),
                               retain_graph=True, create_graph=True)[0][:,2:3]
    
    u_theta = grads_f[:,0:1]
    u_phi   = grads_f[:,1:2]
    u_theta_theta = torch.autograd.grad(u_theta, X_f,
                                        grad_outputs=torch.ones_like(u_theta),
                                        retain_graph=True, create_graph=True)[0][:,0:1]
    u_phi_phi = torch.autograd.grad(u_phi, X_f,
                                    grad_outputs=torch.ones_like(u_phi),
                                    retain_graph=True, create_graph=True)[0][:,1:2]

    sin_theta = torch.sin(theta_f)
    laplace_u = (1/sin_theta)*(torch.cos(theta_f)*u_theta + sin_theta*u_theta_theta) + u_phi_phi/(sin_theta**2)

    f_loss = u_tt - (c/R)**2 * laplace_u
    M_f = torch.mean(f_loss**2)

    # ------------------
    # Initial condition loss
    # ------------------
    X_ic.requires_grad_(True)
    u_ic = model_hat(X_ic)
    theta_ic = X_ic[:,0:1]
    phi_ic   = X_ic[:,1:2]
    u_ic_exact = psi_sphere(theta_ic, phi_ic, theta0, phi0, sigma)

    # initial position/state condition
    loss_u0 = torch.mean((u_ic - u_ic_exact)**2)
    # initial velocity (like in 1D and 2D, we assume the wave starts from rest i.e. v_0=0)
    u_t_ic = torch.autograd.grad(u_ic, X_ic,
                                 grad_outputs=torch.ones_like(u_ic),
                                 retain_graph=True, create_graph=True)[0][:,2:3]
    loss_ut0 = torch.mean(u_t_ic**2)

    M_ic = loss_u0 + loss_ut0

    # ------------------
    # Total loss
    # ------------------
    M_total = beta_f*M_f + beta_ic*M_ic
    return M_total
