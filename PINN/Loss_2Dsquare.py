# u_hat: NN output/estimate: (x,y,t) -> u
# X_f: interior domain collocation points (N_f x 3): (x,y,t)
# X_ic: initial condition domain points (N_ic x 3): (x,y,0)
# X_bc: domain boundary points (N_bc x 3): (x,y,t) , (x,y) in [domain boundary]
# c: wave propagation speed

#torch is imported in the train.py doc, that calls this loss document.
import torch


#our function for computing initial position condition as a gaussian source given x0,y0 defined in train.py
def psi_func(sigma,x0, y0,x,y):
    sigma = 0.2
    return torch.exp(-((x-x0)**2 + (y-y0)**2)/sigma**2)



#the betas are hyperparameters to be tuned and are not found by the PINN during training naturally
def loss_pinn_2D(u_hat, c, beta_f=1.0, beta_ic=1.0, beta_bc=1.0,sigma=1, x0=0,y0=0,
                 x_min=-1.0,x_max=1.0,y_min = -1.0, y_max=1.0, t_min=0.0,t_max=2.0,N=[10000,1000,500]):
    
    #during the training loop we call the criterion i.e. loss function and gives it the following inputs:
        #(outputs, c) = (u_hat, c)
        
    # ------------------
    # Define points in respetivelly; interior domain, initial t=0 interior domain & boundary domain
    # ------------------
    N_f,N_ic,N_bc = N  # interior PDE collocation points
    
    x_f = (x_max - x_min) * torch.rand(N_f, 1) + x_min
    y_f = (y_max - y_min) * torch.rand(N_f, 1) + y_min
    t_f = (t_max - t_min) * torch.rand(N_f, 1) + t_min
    X_f = torch.cat([x_f, y_f, t_f], dim=1)

    x_ic = (x_max - x_min) * torch.rand(N_ic, 1) + x_min
    y_ic = (y_max - y_min) * torch.rand(N_ic, 1) + y_min
    t_ic = torch.zeros(N_ic, 1)  
    X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
    
    # Left (x = x_min) and Right (x = x_max) edges
    N_bc=N_bc_side
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
    
    
    # ------------------
    # PDE loss (interior points)
    # ------------------
    X_f.requires_grad_(True)
    u_f = u_hat(X_f)  # (N_f,1)
    
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
    #middelværdi findes ved at midle over denne sum af l2-normen
    M_f = torch.mean(f_loss**2)
    
    # ------------------
    # Initial condition loss (here we just split it into 2 sums and then sums these 2 sums operators.)
    # ------------------
    X_ic.requires_grad_(True)
    u_ic = u_hat(X_ic)
    u_ic_exact = psi_func(sigma,x0,y0,X_ic[:,0:1], X_ic[:,1:2])  
    
    # Initial position loss
    loss_u0 = torch.mean((u_ic - u_ic_exact)**2)
    # Initial velocity loss
    u_t_ic = torch.autograd.grad(u_ic, X_ic, grad_outputs=torch.ones_like(u_ic),
                                 retain_graph=True, create_graph=True)[0][:,2:3]
    loss_ut0 = torch.mean(u_t_ic**2)
    
    M_ic = loss_u0 + loss_ut0
    
    # ------------------
    # Boundary Neumann loss (in the report we call the N_G, but here we call these points N_bc)
    # ------------------
    X_bc.requires_grad_(True)
    u_bc = u_hat(X_bc)
    
    #differentiate wrt. resptivelly x and y
    u_x_bc = torch.autograd.grad(u_bc, X_bc, grad_outputs=torch.ones_like(u_bc),
                                 retain_graph=True, create_graph=True)[0][:,0:1]
    u_y_bc = torch.autograd.grad(u_bc, X_bc, grad_outputs=torch.ones_like(u_bc),
                                 retain_graph=True, create_graph=True)[0][:,1:2]
    
    # penalize all derivatives at boundaries for simplicity
    M_bc = torch.mean(u_x_bc**2 + u_y_bc**2)
    
    
    
    # ------------------
    # Total loss
    # ------------------
    M_total = beta_f*M_f + beta_ic*M_ic + beta_bc*M_bc
    return M_total
