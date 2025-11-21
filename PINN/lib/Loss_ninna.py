import torch

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
