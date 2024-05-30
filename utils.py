import torch
import numpy as np

def compute_accumulated_transmittance(alphas):
    """
    Compute the accumulated transmittance for a sequence of alpha values.
    
    Parameters:
    alphas (torch.Tensor): Alpha values with shape [batch_size, nb_bins].

    Returns:
    torch.Tensor: Accumulated transmittance with shape [batch_size, nb_bins].
    """
    # Compute the cumulative product of alpha values along the sequence
    accumulated_transmittance = torch.cumprod(alphas, 1)
    # Concatenate a column of ones at the beginning to account for the initial transmittance
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    """
    Render rays using the NeRF model.

    Parameters:
    nerf_model (nn.Module): The NeRF model.
    ray_origins (torch.Tensor): Ray origins with shape [batch_size, 3].
    ray_directions (torch.Tensor): Ray directions with shape [batch_size, 3].
    hn (float, optional): Near bound for ray sampling. Default is 0.
    hf (float, optional): Far bound for ray sampling. Default is 0.5.
    nb_bins (int, optional): Number of bins for sampling along each ray. Default is 192.

    Returns:
    torch.Tensor: Rendered pixel values with shape [batch_size, 3].
    """
    device = ray_origins.device

    # Generate sampling points along each ray
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    
    # Perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]

    # Compute the intervals between adjacent samples
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]

    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    # Pass the 3D points and ray directions through the NeRF model to get colors and densities
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    # Compute alpha values from densities
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]

    # Compute the weights for color accumulation along each ray
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    
    # Regularization for white background
    weight_sum = weights.sum(-1).sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1)
