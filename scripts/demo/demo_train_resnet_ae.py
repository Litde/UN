"""
Usage
-----
python scripts/demo/demo_train_resnet_ae.py --splits ./splits/wikiart_full_inpaint \
    --cache-dir ./data/hf_cache --out ./runs/ae_5pct --epochs 1 --batch-size 16
"""
from __future__ import annotations

import argparse
import os
default_workers = 0 if os.name == "nt" else 4

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image

from database import ArtDatabase
from autoencoders.resnet_autoencoder import ResNetAutoEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=str, required=True, help="Path to saved DatasetDict splits.")
    p.add_argument("--cache-dir", type=str, default="./data/hf_cache", help="HF cache directory.")
    p.add_argument("--out", type=str, default="./runs/ae_5pct", help="Where to save logs and samples.")
    p.add_argument("--size", type=int, default=256, help="Square resize/crop size.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--pct", type=float, default=0.05, help="Fraction of train to use (0-1).")
    p.add_argument("--workers", type=int, default=default_workers)
    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Output dirs
    out_dir = Path(args.out)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Load splits
    db = ArtDatabase(task="inpainting", cache_dir=args.cache_dir)
    db.load_splits(args.splits)

    # --- 5% subsample of TRAIN ---
    train_hf = db.get_train()
    n = int(len(train_hf) * args.pct)
    if n < 1:
        raise SystemExit("Too few samples for the requested percentage.")
    train_hf = train_hf.shuffle(seed=args.seed).select(range(n))

    # Torch adapter + transforms
    transform = T.Compose([
        T.Resize(args.size),
        T.CenterCrop(args.size),
        T.ToTensor(),
    ])
    train_ds = db.as_torch(split="train", transform=transform)
    # override the underlying HF dataset of the wrapper to our 5% subset
    train_ds.base = train_hf  # type: ignore

    def collate(batch):
        # batch: list of (image_tensor, labels_dict)
        imgs = [b[0] for b in batch]
        return torch.stack(imgs, dim=0)

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetAutoEncoder(name=args.backbone, pretrained=True, output_stride=32,
                              use_skips=True, out_channels=3, base_width=256, num_ups=5)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for imgs in loader:
            imgs = imgs.to(device)
            recon, _ = model(imgs)
            loss = loss_fn(recon, imgs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if global_step % 50 == 0:
                print(f"[epoch {epoch}] step {global_step} | L1: {loss.item():.4f}")
            if global_step % 200 == 0:
                # Save a small grid of originals vs reconstructions
                grid = torch.cat([imgs[:8], recon[:8]], dim=0)
                save_image(grid, out_dir / "samples" / f"step_{global_step:06d}.png", nrow=8)
            global_step += 1

    # save final checkpoint
    ckpt_path = out_dir / "ae_last.pt"
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
