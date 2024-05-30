import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import render_rays
from losses import mse, psnr, ssim
import time 

@torch.no_grad()
def test(nerf_model, dataset, hn, hf, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400, device='cpu'):
    """
    Test the NeRF model on a specific image from the dataset.

    Args:
        nerf_model (nn.Module): The NeRF model.
        dataset (torch.Tensor): Dataset containing ray origins, directions, and ground truth colors.
        hn (float): Near plane distance.
        hf (float): Far plane distance.
        chunk_size (int, optional): Chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): Image index to render. Defaults to 0.
        nb_bins (int, optional): Number of bins for density estimation. Defaults to 192.
        H (int, optional): Image height. Defaults to 400.
        W (int, optional): Image width. Defaults to 400.
        device (str, optional): Device to run the computation on. Defaults to 'cpu'.

    Returns:
        None
    """
    # Extract ray origins and directions for the current image
    start_time=time.time()
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    ground_truth_image = dataset[img_index * H * W: (img_index + 1) * H * W, 6:].reshape(H, W, 3)

    data = []  # List to hold regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):  # Iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        
        # Render the rays using the NeRF model
        regenerated_px_values = render_rays(nerf_model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)

    # Concatenate and reshape the data to form an image
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    end_time=time.time()
    # Calculate render time in seconds
    render_time = end_time - start_time


    # Clamp the values to be in the range [0, 1]
    img = np.clip(img, 0, 1)

    # Convert ground truth and rendered images to tensors with shape (1, 3, H, W)
    ground_truth_image = torch.tensor(ground_truth_image).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    rendered_image = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Compute metrics
    mse_value = mse(rendered_image, ground_truth_image).item()
    psnr_value = psnr(rendered_image, ground_truth_image).item()
    ssim_value = ssim(rendered_image, ground_truth_image).item()

    # Print the computed metrics
    print(f"Image {img_index}: MSE: {mse_value}, PSNR: {psnr_value}, SSIM: {ssim_value} , Render Time: {render_time:.4f} seconds")

    # Plot and save the rendered image
    plt.figure()
    plt.imshow(img)
    plt.title(f"Image {img_index}: MSE: {mse_value:.4f}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")
    plt.savefig(f'/kaggle/working/img_{img_index}.png', bbox_inches='tight')
    plt.close()

def train(nerf_model, optimizer, scheduler, data_loader, testing_dataset, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400):
    """
    Train the NeRF model.

    Args:
        nerf_model (nn.Module): The NeRF model.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        testing_dataset (torch.Tensor): Dataset for testing.
        device (str, optional): Device to run the computation on. Defaults to 'cpu'.
        hn (float, optional): Near plane distance. Defaults to 0.
        hf (float, optional): Far plane distance. Defaults to 1.
        nb_epochs (int, optional): Number of epochs to train. Defaults to 100000.
        nb_bins (int, optional): Number of bins for density estimation. Defaults to 192.
        H (int, optional): Image height. Defaults to 400.
        W (int, optional): Image width. Defaults to 400.

    Returns:
        List[float]: Training loss recorded over epochs.
    """
    training_loss = []
    for epoch in range(nb_epochs):
        for batch in tqdm(data_loader):
            # Extract ray origins, directions, and ground truth pixel values from batch
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            # Render the rays using the NeRF model
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            
            # Compute the loss between the ground truth and regenerated pixel values
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss
            training_loss.append(loss.item())
        
        # Step the learning rate scheduler
        scheduler.step()

        # Test the model on a few images
        for img_index in range(3):
            test(nerf_model, testing_dataset, hn, hf, img_index=img_index, nb_bins=nb_bins, H=H, W=W, device=device)
    
    return training_loss
