from basic_inpainting import BasicInpainting, import_params
from dataset import ImageMaskDataset
import torch
from torch import nn
import os
import json

images_dir = '../output/images'
masks_dir = '../output/masks'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epochs: int = 10, params_path: str = 'best_trial.json') -> None:
    params = import_params(params_path)
    model = BasicInpainting(**params)

    # allow missing masks by default (require_mask=False)
    dataset = ImageMaskDataset(images_dir=images_dir, masks_dir=masks_dir, image_size=256, augment=True, require_mask=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.MSELoss()

    # simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(DEVICE)
    for epoch in range(epochs):
        train_loss = model.train_one_epoch(dataloader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")

    model.save_weights('trained_model.pth')



if __name__ == "__main__":
    train(10)