import cv2
import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import re


if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')


def prepare_input(image_pth: str, debug: bool = False) -> Tensor:
    img = cv2.imread(image_pth)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_pth}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1).copy()

    if debug:
        print(f"Prepared input shape: {img.shape}, dtype: {img.dtype}, min/max: {img.min()}/{img.max()}")

    tensor = torch.from_numpy(img)
    if torch.cuda.is_available():
        return tensor.contiguous().pin_memory()
    return tensor.contiguous()


class ImageDataset(Dataset):
    IMAGE_EXTS = ('.png', '.jpg', '.jpeg')

    def __init__(self, corrupted_dir: str, original_dir: str, split: str = 'train'):
        self.corrupted_dir = corrupted_dir
        self.original_dir = original_dir
        self.split = split
        self.samples: List[Tuple[str, str]] = []
        self.num_workers = 0
        self._build_index()

    @staticmethod
    def _list_image_files(dirpath: str) -> List[str]:
        return sorted(
            [f for f in os.listdir(dirpath) if f.lower().endswith(ImageDataset.IMAGE_EXTS)]
        )

    @staticmethod
    def _extract_id(filename: str) -> int | None:
        """
        Extract numeric id from filename.
        Works for:
        - original images: wikiart_1234.jpg, 000045.png
        - corrupted images: corrupted_1234.png
        """
        # try 'corrupted_123' pattern first for clarity
        m = re.search(r"corrupted[_\-]?(\d+)", filename)
        if m:
            return int(m.group(1))

        # fallback: any digits
        m = re.search(r"(\d+)", filename)
        return int(m.group(1)) if m else None

    def _build_index(self) -> None:
        original_files = self._list_image_files(self.original_dir)
        original_map: dict[int, str] = {}
        duplicates = set()

        for f in original_files:
            id_ = self._extract_id(f)
            if id_ is None:
                continue
            if id_ in original_map:
                duplicates.add(id_)
                continue
            original_map[id_] = f

        corrupted_files = self._list_image_files(self.corrupted_dir)

        def corrupted_key(fn):
            id_ = self._extract_id(fn)
            return (id_ is None, id_ if id_ is not None else fn)

        corrupted_files = sorted(corrupted_files, key=corrupted_key)

        progress = tqdm(corrupted_files, desc=f"Indexing {self.split} dataset", unit="file")
        for corrupted_file in progress:
            corrupted_num = self._extract_id(corrupted_file)
            if corrupted_num is None:
                continue

            original_file = original_map.get(corrupted_num)
            if original_file:
                corrupted_path = os.path.join(self.corrupted_dir, corrupted_file)
                original_path = os.path.join(self.original_dir, original_file)
                self.samples.append((corrupted_path, original_path))
        progress.close()

        if not self.samples:
            raise AssertionError("Dataset index is empty \- no matched corrupted/original pairs found")

        if duplicates:
            print(f"Warning: duplicate original IDs found and first occurrence used: {sorted(list(duplicates))}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        corrupted_path, original_path = self.samples[idx]
        corrupted_img = prepare_input(corrupted_path)
        original_img = prepare_input(original_path)
        return corrupted_img, original_img


class ImageDatasetLightning(LightningDataModule):
    def __init__(self, corrupted_dir: str, original_dir: str, batch_size: int = 32):
        super().__init__()
        self.corrupted_dir = corrupted_dir
        self.original_dir = original_dir
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_workers = 0

    def setup(self, stage=None, train_ratio: float = 0.7, val_ratio: float = 0.1):
        full = ImageDataset(self.corrupted_dir, self.original_dir)
        n = len(full)
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
