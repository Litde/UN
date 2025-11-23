import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class ImageMaskDataset(Dataset):
    """Minimalny dataset obraz-mask.

    - Szuka plików w `images_dir` i `masks_dir`.
    - Dopasowuje pliki po nazwie bazowej (bez rozszerzenia).
    - Jeśli `require_mask=False`, obrazy bez maski dostają czarną maskę.
    """

    def __init__(self, images_dir: str, masks_dir: str, image_size: int = 256, require_mask: bool = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.require_mask = require_mask

        img_exts = ('.png', '.jpg', '.jpeg')
        mask_exts = ('.png', '.jpg', '.jpeg', '.npy')

        imgs = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(img_exts)])
        masks = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f)) and f.lower().endswith(mask_exts)])

        img_map = {os.path.splitext(f)[0]: f for f in imgs}
        mask_map = {os.path.splitext(f)[0]: f for f in masks}

        common = sorted(set(img_map.keys()) & set(mask_map.keys()))

        if require_mask and len(common) == 0:
            raise ValueError(f"No matching image-mask pairs found in '{images_dir}' and '{masks_dir}'")

        if require_mask:
            keys = common
        else:
            keys = sorted(img_map.keys())

        self.image_files = [img_map[k] for k in keys]
        self.mask_files = [mask_map.get(k, None) for k in keys]

        self.transform_img = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_file = self.mask_files[idx]
        mask_path = os.path.join(self.masks_dir, mask_file) if mask_file is not None else None

        img = Image.open(img_path).convert('RGB')

        if mask_path is None:
            mask_img = Image.new('L', img.size, 0)
        else:
            if mask_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_img = Image.open(mask_path).convert('L')
            else:
                mask_np = np.load(mask_path)
                if mask_np.dtype != np.uint8:
                    mask_np = (mask_np * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)

        img_t = self.transform_img(img)
        mask_img = self.transform_mask(mask_img)
        mask_t = transforms.ToTensor()(mask_img)
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t

    def split_ds(self, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1, seed=42, shuffle=True):
        n = len(self)
        indices = list(range(n))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        tr = int(train_ratio * n)
        te = tr + int(test_ratio * n)
        train = torch.utils.data.Subset(self, indices[:tr])
        test = torch.utils.data.Subset(self, indices[tr:te])
        valid = torch.utils.data.Subset(self, indices[te:]) if valid_ratio > 0 else None
        return train, test, valid
