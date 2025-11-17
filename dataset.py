import os

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class ImageMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=256, augment=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])
        assert len(self.image_files) == len(self.mask_files), "Images and Masks count mismatch"
        self.image_size = image_size
        self.augment = augment

        self.transform_img = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        img = Image.open(img_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L') if mask_path.lower().endswith(('.png', '.jpg')) else None
        if mask_img is None:
            # load numpy mask
            mask_np = np.load(mask_path)
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))

        img_t = self.transform_img(img)
        mask_img = self.transform_mask(mask_img)
        mask_t = transforms.ToTensor()(mask_img)
        # binary
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t
