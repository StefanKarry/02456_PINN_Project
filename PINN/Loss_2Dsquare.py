import torch

# u_hat: neural network output, takes (x,y,t) -> u
# X_f: interior collocation points (N_f x 3) [x,y,t]
# X_ic: initial condition points (N_ic x 3) [x,y,0]
# X_bc: boundary points (N_bc x 3) [x,y,t]
# c: wave speed

def psi_func(x,y):
    sigma = 0.2
    return torch.exp(-((x-x0)**2 + (y-y0)**2)/sigma**2)


def loss_pinn_2D(u_hat, X_f, X_ic, psi_func, X_bc, beta_f=1.0, beta_ic=1.0, beta_bc=1.0):
    # ------------------
    # PDE loss (interior points)
    # ------------------
    X_f.requires_grad_(True)
    u_f = u_hat(X_f)  # [N_f,1]
    
    # Compute derivatives
    u_t = torch.autograd.grad(u_f, X_f, grad_outputs=torch.ones_like(u_f), 
                              retain_graph=True, create_graph=True)[0][:,2:3]
    u_tt = torch.autograd.grad(u_t, X_f, grad_outputs=torch.ones_like(u_t),
                               retain_graph=True, create_graph=True)[0][:,2:3]
    
    u_x = torch.autograd.grad(u_f, X_f, grad_outputs=torch.ones_like(u_f),
                              retain_graph=True, create_graph=True)[0][:,0:1]
    u_xx = torch.autograd.grad(u_x, X_f, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:,0:1]
    
    u_y = torch.autograd.grad(u_f, X_f, grad_outputs=torch.ones_like(u_f),
                              retain_graph=True, create_graph=True)[0][:,1:2]
    u_yy = torch.autograd.grad(u_y, X_f, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:,1:2]
    
    f_loss = u_tt - c**2 * (u_xx + u_yy)
    M_f = torch.mean(f_loss**2)
    
    # ------------------
    # Initial condition loss
    # ------------------
    X_ic.requires_grad_(True)
    u_ic = u_hat(X_ic)
    u_ic_exact = psi_func(X_ic[:,0:1], X_ic[:,1:2])  # psi(x,y)
    
    # Initial position loss
    loss_u0 = torch.mean((u_ic - u_ic_exact)**2)
    
    # Initial velocity loss
    u_t_ic = torch.autograd.grad(u_ic, X_ic, grad_outputs=torch.ones_like(u_ic),
                                 retain_graph=True, create_graph=True)[0][:,2:3]
    loss_ut0 = torch.mean(u_t_ic**2)
    
    M_ic = loss_u0 + loss_ut0
    
    # ------------------
    # Boundary Neumann loss
    # ------------------
    X_bc.requires_grad_(True)
    u_bc = u_hat(X_bc)
    
    u_x_bc = torch.autograd.grad(u_bc, X_bc, grad_outputs=torch.ones_like(u_bc),
                                 retain_graph=True, create_graph=True)[0][:,0:1]
    u_y_bc = torch.autograd.grad(u_bc, X_bc, grad_outputs=torch.ones_like(u_bc),
                                 retain_graph=True, create_graph=True)[0][:,1:2]
    
    # For square domain: pick the correct component based on boundary location
    # If X_bc[:,0] is at left/right -> derivative w.r.t x, else derivative w.r.t y
    # Here we just penalize all derivatives at boundaries for simplicity
    M_bc = torch.mean(u_x_bc**2 + u_y_bc**2)
    
    # ------------------
    # Total loss
    # ------------------
    M_total = beta_f*M_f + beta_ic*M_ic + beta_bc*M_bc
    return M_total, M_f, M_ic, M_bc
