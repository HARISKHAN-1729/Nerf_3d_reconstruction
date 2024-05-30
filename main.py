import torch
import numpy as np
from torch.utils.data import DataLoader
from nerf import NerfModel
from train import train

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    training_dataset = torch.from_numpy(np.load('/kaggle/input/nerf-train-test/training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('/kaggle/input/nerf-train-test/testing_data.pkl', allow_pickle=True))

    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader, testing_dataset, nb_epochs=16, device=device, hn=2, hf=6, nb_bins=192, H=400, W=400)
