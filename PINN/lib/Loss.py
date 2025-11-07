import torch

def wave_square_loss(prediction, target, c=1.4):
    """
    Compute the Wave Square Loss between prediction and target.

    Args:
        prediction (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground truth values.
        c (float): Wave speed constant.

    Returns:
        float: The computed Wave Square Loss.
    """
    x = prediction[:, 0]
    t = prediction[:, 1]
    # Compute the wave square loss
    domain_loss = torch.mean((torch.autograd.grad(torch.autograd.grad(prediction, t), t)- 
                             c**2*torch.autograd.grad(torch.autograd.grad(prediction, x), x))**2)
    IC_loss = torch.mean((prediction[:, 0] - target[:,0])**2+
                         (torch.autograd.grad(prediction, t)[:,0])**2)
    boundary_loss = torch.mean(torch.autograd.grad(prediction, x)**2)
    return domain_loss + IC_loss + boundary_loss