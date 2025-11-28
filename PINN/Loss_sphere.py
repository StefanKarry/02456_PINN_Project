import torch
from torch.utils.data import Dataset
import numpy as np

# -------------------
# Initial Gaussian on the sphere (for t=0 condition)
# -------------------
def psi_sphere(theta, phi, theta0=1.0, phi0=1.0, sigma=0.2):
    return torch.exp(-((theta-theta0)**2 + (phi-phi0)**2)/sigma**2)


# -----------------------------
# Wave dataset for 3D spherical domain
# -----------------------------
class WaveDataset(Dataset):
    def __init__(self, num_samples=1000, train=True,
                 t_min=0.0, t_max=2.0, theta0=1.0, phi0=1.0, sigma=0.2):
        super().__init__()
        self.num_samples = num_samples
        self.t_min = t_min
        self.t_max = t_max
        self.theta0 = theta0
        self.phi0 = phi0
        self.sigma = sigma

        # Sample points randomly on the sphere and in time
        # These are independent of collocation points used in PINN loss
        self.theta = torch.rand(num_samples, 1) * np.pi        # [0, pi]
        self.phi = torch.rand(num_samples, 1) * 2*np.pi        # [0, 2pi]
        self.t = t_min + (t_max - t_min) * torch.rand(num_samples, 1)

        # Compute target u only at t=0 if you want initial condition
        # For generalization, we can either use u=0 for t>0, or analytical solution if available
        self.u = psi_sphere(self.theta, self.phi, theta0, phi0, sigma)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return input (theta, phi, t) and target u
        input_xyz = torch.cat([self.theta[idx:idx+1], self.phi[idx:idx+1], self.t[idx:idx+1]], dim=1)
        target_u = self.u[idx:idx+1]
        return input_xyz, target_u



# -------------------
# Spherical PINN Loss
# -------------------
def loss_pinn_sphere(u_hat, c, beta_f=1.0, beta_ic=1.0, sigma=0.2,theta0=1.0, phi0=1.0, R=1.0,
                     t_min=0.0, t_max=2.0, N=[10000,1000]):
    
    N_f, N_ic = N  # number of interior and initial points (N_ic[theta,phi]=N_f[theta,phi,t=0])

    # ------------------
    # 1. Sample points (theta, phi, t)
    # ------------------
    theta_f = torch.rand(N_f,1) * np.pi    #theta: forms and angle with the z-axis i.e. in [0,pi]
    phi_f   = torch.rand(N_f,1) * 2*np.pi  #phi: forms an angle with the x axis in the xy plane i.e. in [0,2pi]
    t_f     = t_min + (t_max - t_min) * torch.rand(N_f,1)  #using our default tmin and tmax inputs
    X_f = torch.cat([theta_f, phi_f, t_f], dim=1)

    theta_ic = torch.rand(N_ic,1) * np.pi
    phi_ic   = torch.rand(N_ic,1) * 2*np.pi
    t_ic     = torch.zeros(N_ic,1)
    X_ic = torch.cat([theta_ic, phi_ic, t_ic], dim=1)

    # ------------------
    # PDE Loss (interior points)
    # ------------------
    X_f.requires_grad_(True)  #activate so we can compute the gradients 
    u_f = u_hat(X_f)

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
    u_ic = u_hat(X_ic)
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
