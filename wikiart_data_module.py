from datasets import load_dataset
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import pytorch_lightning as pl
import torch
from tqdm import tqdm

class WikiArtSRDataset(Dataset):
    def __init__(self, hf_dataset, scale=2, crop_size=41):
        self.dataset = hf_dataset
        self.scale = scale
        self.crop_size = crop_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"].convert("YCbCr")
        y, _, _ = img.split()

        w, h = y.size
        if w < self.crop_size or h < self.crop_size:
            y = y.resize((self.crop_size, self.crop_size))

        x = torch.randint(0, y.width - self.crop_size + 1, (1,)).item()
        y0 = torch.randint(0, y.height - self.crop_size + 1, (1,)).item()

        hr = y.crop((x, y0, x + self.crop_size, y0 + self.crop_size))

        lr = hr.resize(
            (self.crop_size // self.scale, self.crop_size // self.scale),
            Image.BICUBIC
        )
        ilr = lr.resize(hr.size, Image.BICUBIC)

        return TF.to_tensor(ilr), TF.to_tensor(hr)

class WikiArtFullImageDataset(Dataset):
    def __init__(self, hf_dataset, scale=2):
        self.dataset = hf_dataset
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"].convert("YCbCr")
        y, _, _ = img.split()

        w, h = y.size
        w -= w % self.scale
        h -= h % self.scale
        y = y.crop((0, 0, w, h))

        hr = y
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        ilr = lr.resize((w, h), Image.BICUBIC)

        return TF.to_tensor(ilr), TF.to_tensor(hr)

class CachedWikiArtFullImageDataset(Dataset):
    def __init__(self, hf_dataset, scale=2, max_images=50):
        self.samples = []
        self.scale = scale

        # Limit validation size
        hf_dataset = hf_dataset.select(range(max_images))

        print(f"Caching {len(hf_dataset)} validation images...")

        for item in tqdm(hf_dataset, desc="Preparing validation set"):
            img = item["image"].convert("YCbCr")
            y, _, _ = img.split()

            w, h = y.size
            w -= w % scale
            h -= h % scale
            y = y.crop((0, 0, w, h))

            hr = TF.to_tensor(y)

            lr = y.resize((w // scale, h // scale), Image.BICUBIC)
            ilr = lr.resize((w, h), Image.BICUBIC)
            ilr = TF.to_tensor(ilr)

            self.samples.append((ilr, hr))

        print("Validation images cached in memory.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class WikiArtDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, scale=2):
        super().__init__()
        self.batch_size = batch_size
        self.scale = scale

    def setup(self, stage=None):
        full_ds = load_dataset("huggan/wikiart", split="train")

        split_ds = full_ds.train_test_split(
            test_size=0.05,
            seed=42
        )

        self.train_ds = WikiArtSRDataset(
            split_ds["train"],
            scale=self.scale
        )

        self.val_ds = CachedWikiArtFullImageDataset(
            split_ds["test"],
            scale=self.scale,
            max_images=50
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,  # full-image inference
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
