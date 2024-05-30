import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import render_rays
from losses import mse, psnr, ssim

@torch.no_grad()
def test(nerf_model, dataset, hn, hf, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400, device='cpu'):
    """
    Args:
        hn: near plane distance
        hf: far plane distance
        dataset: dataset to render
        chunk_size (int, optional): chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): image index to render. Defaults to 0.
        nb_bins (int, optional): number of bins for density estimation. Defaults to 192.
        H (int, optional): image height. Defaults to 400.
        W (int, optional): image width. Defaults to 400.

    Returns:
        None: None
    """
    # Extract ray origins and directions for the current image
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    ground_truth_image = dataset[img_index * H * W: (img_index + 1) * H * W, 6:].reshape(H, W, 3)

    data = []   # List to hold regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # Iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        regenerated_px_values = render_rays(nerf_model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)

    # Concatenate and reshape the data to form an image
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    # Clamp the values to be in the range [0, 1]
    img = np.clip(img, 0, 1)

    # Convert ground truth and rendered images to tensors with shape (1, 3, H, W)
    ground_truth_image = torch.tensor(ground_truth_image).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)
    rendered_image = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)

    # Compute metrics
    mse_value = mse(rendered_image, ground_truth_image).item()
    psnr_value = psnr(rendered_image, ground_truth_image).item()
    ssim_value = ssim(rendered_image, ground_truth_image).item()

    print(f"Image {img_index}: MSE: {mse_value}, PSNR: {psnr_value}, SSIM: {ssim_value}")

    # Plot and save the image
    plt.figure()
    plt.imshow(img)
    plt.title(f"Image {img_index}: MSE: {mse_value:.4f}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")
    plt.savefig(f'/kaggle/working//img_{img_index}.png', bbox_inches='tight')
    plt.close()

def train(nerf_model, optimizer, scheduler, data_loader, testing_dataset, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400):
    training_loss = []
    for _ in range(nb_epochs):
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()

        for img_index in range(3):
            test(nerf_model, testing_dataset, hn, hf, img_index=img_index, nb_bins=nb_bins, H=H, W=W, device=device)
    return training_loss
