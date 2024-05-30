import torch
from kornia.losses import SSIMLoss

dssim_loss = SSIMLoss(3)

import torch

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    """
    Computes the Mean Squared Error (MSE) between the predicted image and the ground truth image.
    
    Parameters:
    image_pred (torch.Tensor): The predicted image tensor.
    image_gt (torch.Tensor): The ground truth image tensor.
    valid_mask (torch.Tensor, optional): A boolean tensor indicating valid pixels to consider in the computation. Default is None.
    reduction (str, optional): Specifies the reduction to apply to the output. 'mean' will return the average of the values, 
                               any other value will return the raw squared differences. Default is 'mean'.
    
    Returns:
    torch.Tensor: The computed MSE value if reduction is 'mean', otherwise the tensor of squared differences.
    """
    
    # Compute the element-wise squared differences between the predicted and ground truth images
    squared_diff = (image_pred - image_gt) ** 2
    
    # If a valid_mask is provided, use it to filter the squared differences
    if valid_mask is not None:
        squared_diff = squared_diff[valid_mask]
    
    # If reduction is 'mean', compute and return the mean of the squared differences
    if reduction == 'mean':
        return torch.mean(squared_diff)
    
    # If no reduction is specified, return the tensor of squared differences
    return squared_diff


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the predicted image and the ground truth image.
    
    Parameters:
    image_pred (torch.Tensor): The predicted image tensor.
    image_gt (torch.Tensor): The ground truth image tensor.
    valid_mask (torch.Tensor, optional): A boolean tensor indicating valid pixels to consider in the computation. Default is None.
    reduction (str, optional): Specifies the reduction to apply to the MSE calculation. 'mean' will return the average MSE value, 
                               any other value will return the raw MSE value. Default is 'mean'.
    
    Returns:
    torch.Tensor: The computed PSNR value.
    """
    
    # Compute the Mean Squared Error (MSE) using the mse function
    mse_value = mse(image_pred, image_gt, valid_mask, reduction)
    
    # Compute the PSNR value using the MSE
    psnr_value = -10 * torch.log10(mse_value)
    
    return psnr_value

import torch

def ssim(image_pred, image_gt, reduction='mean'):
    """
    Computes the Structural Similarity Index (SSIM) between the predicted image and the ground truth image.
    
    Parameters:
    image_pred (torch.Tensor): The predicted image tensor with shape (1, 3, H, W).
    image_gt (torch.Tensor): The ground truth image tensor with shape (1, 3, H, W).
    reduction (str, optional): Specifies the reduction to apply to the output of the dssim_loss function. 
                               'mean' will return the average value, any other value will return the raw value. Default is 'mean'.
    
    Returns:
    torch.Tensor: The computed SSIM value.
    """
    
    # Compute the dissimilarity using the dssim_loss function, which returns a value in the range [0, 1]
    dssim_value = dssim_loss(image_pred, image_gt)
    
    # Compute the SSIM value from the dissimilarity, scaling it to be in the range [-1, 1]
    ssim_value = 1 - 2 * dssim_value
    
    return ssim_value

