import cv2
import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import List, Tuple
from tqdm import tqdm

def prepare_input(image_pth: str, debug:bool = False) -> Tensor:
    img = cv2.imread(image_pth)
    img = cv2.resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img.reshape(3, 256, 256)
    if debug:
        print(f"Prepared input shape: {img.shape}")
    return torch.tensor(img).float()

class ImageDataset(Dataset):
    def __init__(self, corrupted_dir: str, original_dir: str, split: str = 'train'):
        self.corrupted_dir = corrupted_dir
        self.original_dir = original_dir
        self.split = split
        self.dataset: List[Tuple[Tensor, Tensor]] = []
        self._load_data()

    def _load_data(self) -> None:
        files_original = os.listdir(self.original_dir)
        original_map = {}
        for f in files_original:
            num = os.path.splitext(f)[0]  # '0000.jpg' -> '0000'
            original_map[int(num)] = f  # convert to int for matching

        files_corrupted = os.listdir(self.corrupted_dir)
        for corrupted_file in tqdm(files_corrupted, desc=f"Loading {self.split} dataset"):
            corrupted_num_str = os.path.splitext(corrupted_file)[0].replace("corrupted_", "")
            corrupted_num = int(corrupted_num_str)

            if corrupted_num in original_map:
                original_file = original_map[corrupted_num]

                corrupted_path = os.path.join(self.corrupted_dir, corrupted_file)
                original_path = os.path.join(self.original_dir, original_file)

                corrupted_img = prepare_input(corrupted_path)
                original_img = prepare_input(original_path)

                self.dataset.append((corrupted_img, original_img))
        assert len(self.dataset) != 0, f"Dataset is empty"


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return self.dataset[idx]


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
        """
            Create train, val, test splits.
            Also, it automatically splits test datase based on the two given ratios. (train_ratio, val_ratio)
            So the test_ratio will be 1 - train_ratio - val_ratio.
        """
        full = ImageDataset(self.corrupted_dir, self.original_dir)
        n = len(full)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        test_n = n - train_n - val_n

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full, [train_n, val_n, test_n])

        self.num_workers = os.cpu_count() - 1 if os.cpu_count() - 1 is not None else 4

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
