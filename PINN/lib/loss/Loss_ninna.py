import torch

def compute_grad(u, x):
    """
    Compute the gradient of u with respect to x.
    Args:
        u (torch.Tensor): The output tensor.
        x (torch.Tensor): The input tensor.
    """
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

#Markus
def waveLoss_2D(model, x_f: torch.Tensor, t_f: torch.Tensor, x_b: torch.Tensor, t_b: torch.Tensor, x_0: torch.Tensor, t_0: torch.Tensor, u_0_target: torch.Tensor, c=1.4, beta_f=1.0, beta_ic=1.0):
    """
    Compute the Wave Loss for 2D wave equation.

    Args:
        model (torch.nn.Module): The neural network model.
        colPoints (torch.Tensor): Collocation points for the domain. [x_f, t_f]
        boundaryPoints (torch.Tensor): Boundary points. [x_b, t_b]
        initialPoints (torch.Tensor): Initial condition points. [x_0, t_0]
        u_0_target (torch.Tensor): Target values for the initial condition.
        c (float): Wave speed constant.
        beta_f (float): Weight for the domain loss.
        beta_ic (float): Weight for the initial condition loss.
    Returns:
        float: The computed Wave Loss.
    """ 
    # Predict at collocation points
    u_pred = model(x_f, t_f)

    # Compute derivatives
    u_t = compute_grad(u_pred, t_f)
    u_x = compute_grad(u_pred, x_f)

    # Compute second derivatives
    u_tt = compute_grad(u_t, t_f)
    u_xx = compute_grad(u_x, x_f)
    # Domain loss (PDE residual [u_tt - c^2 * u_xx = 0])
    res = u_tt - c**2 * u_xx
    loss_f = torch.mean(res**2) #Mean Squared Error (MSE) of the resiudal

    # Boundary loss (Neumann BC: u_x(-1,t) = u_x(1,t) = 0)
    u_b_pred = model(x_b, t_b)
    u_b_x = compute_grad(u_b_pred, x_b)
    loss_b = torch.mean(u_b_x**2) #MSE of the boundary condition

    # Initial condition loss
    # Position loss (u(x,0) = f(x))
    u_initial_pred = model(x_0, t_0)
    loss_ic_pos = torch.mean((u_initial_pred - u_0_target) ** 2)

    # Velocity loss (u_t(x,0) = 0)
    u_initial_t_deriv = compute_grad(u_initial_pred, t_0)
    loss_ic_vel = torch.mean(u_initial_t_deriv**2) #MSE of initial velocity condition

    # Total loss
    return beta_f * loss_f + beta_ic * (loss_ic_pos + loss_ic_vel) + loss_b




def wave_square_loss(prediction, target, c=1.4, w_d=1.0, w_ic=1.0, w_bc=1.0):
    """
    Compute the Wave Square Loss between prediction and target.

    Args:
        prediction (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground truth values.
        c (float): Wave speed constant.
        w_d (float): Weight for the domain loss.
        w_ic (float): Weight for the initial condition loss.
        w_bc (float): Weight for the boundary condition loss.

    Returns:
        float: The computed Wave Square Loss.
    """
    x = prediction[:, 0]
    t = prediction[:, 1]
    # Compute the wave square loss
    domain_loss = torch.mean((torch.autograd.grad(torch.autograd.grad(prediction, t), t)- 
                             c**2*torch.autograd.grad(torch.autograd.grad(prediction, x), x))**2)
    IC_loss = torch.mean((x[0] - target[0])**2+
                         (torch.autograd.grad(x[0], t))**2)
    boundary_loss = torch.mean(torch.autograd.grad(prediction, x)**2)

    return w_d * domain_loss + w_ic * IC_loss + w_bc * boundary_loss

def wave_sphere_loss(prediction, target, c=1.4, w_d=1.0, w_ic=1.0, w_bc=1.0):
    """
    Compute the Wave Sphere Loss between prediction and target.

    Args:
        prediction (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground truth values.
        c (float): Wave speed constant.
        w_d (float): Weight for the domain loss.
        w_ic (float): Weight for the initial condition loss.
        w_bc (float): Weight for the boundary condition loss.

    Returns:
        float: The computed Wave Sphere Loss.
    """
    R = prediction[:, 0]
    theta = prediction[:, 1]
    phi = prediction[:, 2]
    t = prediction[:, 3]
    
    # Compute the wave sphere loss
    domain_loss = torch.mean((torch.autograd.grad(torch.autograd.grad(prediction, z), z) -
                             c**2*torch.autograd.grad(torch.autograd.grad(prediction, x), x))**2)
    IC_loss = torch.mean((prediction[:, 0] - target[:,0])**2+
                         (torch.autograd.grad(prediction, z)[:,0])**2)
    boundary_loss = torch.mean(torch.autograd.grad(prediction, x)**2)

    return w_d * domain_loss + w_ic * IC_loss + w_bc * boundary_loss
