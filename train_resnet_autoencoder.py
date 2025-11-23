"""train_resnet_autoencoder.py

Train the ResNetAutoEncoder on the full TRAIN split with validation.
Includes the recommended improvements:
- input normalized to [-1,1] and decoder with final_act="tanh"
- output_stride=16 (bottleneck 16x16) for more detail
- reconstruction loss = 1.0 * L1 + 0.1 * Perceptual(VGG16 relu3_3) + 0.01 * TV
- PSNR/SSIM metrics on validation and best-checkpoint saving (by PSNR)
- fixed evaluation batch to make sample grids comparable across steps

Usage
-----
python scripts/train_resnet_autoencoder.py \
  --splits ./splits/wikiart_full_70_15_15 \
  --cache-dir ./data/hf_cache \
  --out ./runs/resnet_ae_os16 \
  --epochs 10 --batch-size 32 --lr 2e-4 --size 256

Notes
-----
- On Windows, default num_workers=0 (you can raise after making TorchAdapter pickleable).
- If you prefer a tighter bottleneck, set --os 32 (gives 8x8 top) and consider raising base_width.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T, models
from torchvision.utils import save_image

from database import ArtDatabase
from autoencoders.resnet_autoencoder import ResNetAutoEncoder

try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    HAS_TM = True
except Exception:
    HAS_TM = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--cache-dir", default="./data/hf_cache")
    p.add_argument("--out", default="./runs/resnet_ae_os16")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 4))
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18","resnet34","resnet50","resnet101","resnet152"])
    p.add_argument("--os", type=int, default=16, choices=[8,16,32], help="Output stride of encoder")
    p.add_argument("--base-width", type=int, default=256, help="Decoder base width")
    p.add_argument("--save-every", type=int, default=1000, help="Save samples every N steps")
    return p.parse_args()


# ---------------- Perceptual Loss (VGG16 relu3_3) -----------------
class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        # up to relu3_3 -> indices up to 16 (conv-relu blocks)
        self.slice = nn.Sequential(*list(vgg.children())[:17]).eval()
        for p in self.slice.parameters():
            p.requires_grad_(False)
        # ImageNet mean/std in [0,1]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # x,y in [-1,1] -> to [0,1] then ImageNet norm
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        fx = self.slice(x)
        fy = self.slice(y)
        return F.l1_loss(fx, fy)


def total_variation(x):
    return (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean() + (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    # --- Data ---
    db = ArtDatabase(task="inpainting", cache_dir=args.cache_dir)
    db.load_splits(args.splits)

    transform = T.Compose([
        T.Resize(args.size),
        T.CenterCrop(args.size),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # -> [-1,1]
    ])

    train_ds = db.as_torch(split="train", transform=transform)
    val_ds = db.as_torch(split="val", transform=transform)

    def collate(batch):
        return torch.stack([b[0] for b in batch], dim=0)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, persistent_workers=False,
                          pin_memory=torch.cuda.is_available(), collate_fn=collate)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, persistent_workers=False,
                        pin_memory=torch.cuda.is_available(), collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed eval batch for consistent grids
    fixed_eval = next(iter(val_ld)).to(device)

    # --- Model ---
    model = ResNetAutoEncoder(
        name=args.backbone,
        pretrained=True,
        output_stride=args.os,
        use_skips=True,
        out_channels=3,
        base_width=args.base_width,
        num_ups=None,
        final_act="tanh",  # match [-1,1]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    l1 = nn.L1Loss()
    perc = VGGPerceptual().to(device)

    best_psnr = -1.0
    global_step = 0

    for ep in range(args.epochs):
        model.train()
        for imgs in train_ld:
            imgs = imgs.to(device)
            recon, _ = model(imgs)
            loss = l1(recon, imgs) + 0.1 * perc(recon, imgs) + 0.01 * total_variation(recon)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if global_step % 50 == 0:
                print(f"[train] epoch {ep} step {global_step} ...")

            if global_step % args.save_every == 0:
                with torch.no_grad():
                    # log current batch
                    grid = torch.cat([(imgs[:8]+1)/2, (recon[:8]+1)/2], dim=0)
                    save_image(grid, out_dir/"samples"/f"train_step_{global_step:06d}.png", nrow=8)
                    # log fixed eval batch
                    rec_eval, _ = model(fixed_eval)
                    grid_eval = torch.cat([(fixed_eval[:8]+1)/2, (rec_eval[:8]+1)/2], dim=0)
                    save_image(grid_eval, out_dir/"samples"/f"val_fixed_step_{global_step:06d}.png", nrow=8)
            global_step += 1

        # ---- validation epoch end ----
        model.eval()
        with torch.no_grad():
            total_l1 = total_mse = total_psnr = total_ssim = n = 0
            for imgs in val_ld:
                imgs = imgs.to(device)
                recon, _ = model(imgs)
                total_l1 += torch.mean(torch.abs(recon - imgs)).item() * imgs.size(0)
                mse = torch.mean((recon - imgs) ** 2).item()
                total_mse += mse * imgs.size(0)
                # PSNR for [-1,1]: peak-to-peak=2.0
                ps = 10.0 * torch.log10((2.0 ** 2) / max(mse, 1e-12))
                total_psnr += float(ps) * imgs.size(0)
                if HAS_TM:
                    total_ssim += float(ssim((recon+1)/2, (imgs+1)/2, data_range=1.0)) * imgs.size(0)
                n += imgs.size(0)

            l1_val = total_l1 / n
            mse_val = total_mse / n
            psnr_val = total_psnr / n
            ssim_val = (total_ssim / n) if HAS_TM else float('nan')

            print(f"[val] epoch {ep}: L1={l1_val:.4f} MSE={mse_val:.6f} PSNR={psnr_val:.2f} SSIM={ssim_val:.4f}")

            # Save best by PSNR
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                best_path = out_dir / "ae_best.pt"
                torch.save({
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": ep,
                    "psnr": best_psnr,
                }, best_path)
                print(f"[val] new best (PSNR={best_psnr:.2f}) saved to {best_path}")

        sched.step()

    # save last
    last_path = out_dir / "ae_last.pt"
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": args.epochs}, last_path)
    print(f"Saved last checkpoint to: {last_path}")


if __name__ == "__main__":
    main()
