import torch
def compute_grad(u, x):
    """
    Compute the gradient of u with respect to x.
    Args:
        u (torch.Tensor): The output tensor.
        x (torch.Tensor): The input tensor.
    """
    return torch.autograd.grad(u, x,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True)[0]

def waveLoss_1D(model, x_f: torch.Tensor, t_f: torch.Tensor, x_b_r: torch.Tensor, x_b_l: torch.Tensor, t_b: torch.Tensor, x_0: torch.Tensor, t_0: torch.Tensor, u_0_target: torch.Tensor, c=1.4, beta_f=1.0, beta_ic=1.0, beta_b=1.0):
    """
    Compute the Wave Loss for 1D wave equation.

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

    #Checking if the the input tensors require gradients for autograd
    assert x_f.requires_grad, "x_f must require gradients"
    assert t_f.requires_grad, "t_f must require gradients"
    assert x_b_r.requires_grad, "x_b_r must require gradients"
    assert x_b_l.requires_grad, "x_b_l must require gradients"
    assert t_b.requires_grad, "t_b must require gradients"
    assert x_0.requires_grad, "x_0 must require gradients"
    assert t_0.requires_grad, "t_0 must require gradients"

    # Domain loss (PDE residual [u_tt - c^2 * u_xx = 0])
    u_pred = model(x_f, t_f)
    # Compute derivatives
    u_t = compute_grad(u_pred, t_f)
    u_x = compute_grad(u_pred, x_f)

    # Compute second derivatives
    u_tt = compute_grad(u_t, t_f)
    u_xx = compute_grad(u_x, x_f)

    # By recommendation of Gemini, add time weighting to the collocation loss. This gives higher weightings to later points where it intially seemed uncertain.
    time_weight = 1.0 + 3.0 * t_f

    res = u_tt - c**2 * u_xx
    loss_F = torch.mean(time_weight.detach() * res**2) #Mean Squared Error (MSE) of the resiudal

    # Prediction at left and right edges
    # left edge (x = -1)
    u_pred_left = model(x_b_l, t_b)
    u_x_left = compute_grad(u_pred_left, x_b_l)[:, 0:1] # Grad w.r.t x
    loss_b_left = torch.mean(u_x_left**2)

    # right edge (x = 1)
    u_pred_right = model(x_b_r, t_b)
    u_x_right = compute_grad(u_pred_right, x_b_r)[:, 0:1] # Grad w.r.t x
    loss_b_right = torch.mean(u_x_right**2)

    loss_B = loss_b_left + loss_b_right

    # Initial condition loss
    # Position loss (u(x,0) = f(x))
    u_initial_pred = model(x_0, t_0)
    loss_ic_pos = torch.mean((u_initial_pred - u_0_target) ** 2)

    # Velocity loss (u_t(x,0) = 0)
    u_initial_t_deriv = compute_grad(u_initial_pred, t_0)
    loss_ic_vel = torch.mean(u_initial_t_deriv**2) #MSE of initial velocity condition

    loss_IC = loss_ic_pos + loss_ic_vel

    # Total loss
    return beta_f*loss_F + beta_ic*loss_IC + beta_b*loss_B
