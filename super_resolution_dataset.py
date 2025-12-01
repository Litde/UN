import cv2
import numpy as np
import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
from typing import List, Tuple
import re
from utils import image_resize, bgr2ycbcr,image2tensor

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

def prepare_input(image_pth: str, debug: bool = False):
    image = cv2.imread(image_pth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    hr_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    lr_image = image_resize(hr_image, 0.5)
    lr_image = image_resize(lr_image, 2)

    lr_y_image = bgr2ycbcr(lr_image, only_use_y_channel=True)
    hr_y_image = bgr2ycbcr(hr_image, only_use_y_channel=True)

    lr_y_tensor = image2tensor(lr_y_image, range_norm=False, half=False)
    hr_y_tensor = image2tensor(hr_y_image, range_norm=False, half=False)

    return lr_y_tensor, hr_y_tensor

class ImageDataset(Dataset):
    IMAGE_EXTS = ('.png', '.jpg', '.jpeg')

    def __init__(self, image_dir: str, split: str = 'train'):
        self.image_dir = image_dir
        self.split = split
        self.samples: List[str] = []
        self.num_workers = 0
        self._build_index()

    @staticmethod
    def _list_image_files(dirpath: str) -> List[str]:
        return sorted(
            [f for f in os.listdir(dirpath) if f.lower().endswith(ImageDataset.IMAGE_EXTS)]
        )

    @staticmethod
    def _extract_id(filename: str) -> int | None:
        m = re.search(r"(\d+)", filename)
        return int(m.group(1)) if m else None

    def _build_index(self) -> None:
        img_files = self._list_image_files(self.image_dir)

        for f in img_files:
            id_ = self._extract_id(f)
            if id_ is None:
                continue

        progress = tqdm(img_files, desc=f"Indexing {self.split} dataset", unit="file")
        for img_file in progress:
            img_num = self._extract_id(img_file)
            if img_num is None:
                continue

            if img_file:
                original_path = os.path.join(self.image_dir, img_file)
                self.samples.append(original_path)
        progress.close()

        if not self.samples:
            raise AssertionError("Dataset index is empty \- no matched corrupted/original pairs found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        image_dir = self.samples[idx]
        return prepare_input(image_dir)


class ImageDatasetLightning(LightningDataModule):
    def __init__(self, image_dir: str, batch_size: int = 32):
        super().__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_workers = 0

    def setup(self, stage=None, train_ratio: float = 0.7, val_ratio: float = 0.1):
        full = ImageDataset(self.image_dir)
        n = len(full)
        # print(n)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        test_n = n - train_n - val_n

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full, [train_n, val_n, test_n]
        )

        self.num_workers = max(1, (os.cpu_count() or 4) - 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )