"""scripts/train_resnet_autoencoder.py

Train the ResNetAutoEncoder on the full TRAIN split with validation.
Includes the recommended improvements:
- input normalized to [-1,1] and decoder with final_act="tanh"
- output_stride=16 (bottleneck 16x16) for more detail
- reconstruction loss = 1.0 * L1 + 0.1 * Perceptual(VGG16 relu3_3) + 0.01 * TV
- PSNR/SSIM metrics on validation and best-checkpoint saving (by PSNR)
- fixed evaluation batch to make sample grids comparable across steps

Usage
-----
python train_resnet_autoencoder.py \
  --splits ./splits/wikiart_full_70_15_15 \
  --cache-dir ./data/hf_cache \
  --out ./runs/resnet_ae_os16 \
  --epochs 10 --batch-size 32 --lr 2e-4 --size 256 --os 16

or

python train_resnet_autoencoder.py \
  --splits ./splits/wikiart_full_70_15_15 \
  --cache-dir ./data/hf_cache \
  --out ./runs/resnet_ae_os16 \
  --epochs 10 --batch-size 32 --lr 2e-4 --size 256 --os 16 \
  --resume ./runs/resnet_ae_os16/checkpoints/ckpt_step_000200.pt

Notes
-----
- On Windows, default num_workers=0 (you can raise after making TorchAdapter pickleable).
- If you prefer a tighter bottleneck, set --os 32 (gives 8x8 top) and consider raising base_width.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import math
from pathlib import Path

import numpy as np
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
    p.add_argument("--save-every", type=int, default=300, help="Save samples every N steps")
    p.add_argument("--ckpt-every", type=int, default=300, help="Save checkpoint every N steps (0=off)")
    p.add_argument("--ckpt-epoch", action="store_true", help="Also save checkpoint at end of each epoch")
    p.add_argument("--keep-last", type=int, default=3, help="Keep last N step-checkpoints (rolling)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--logger", type=str, default="none", choices=["none","comet","wandb"])
    p.add_argument("--project", type=str, default="wikiart-ae", help="Project name (for comet/wandb)")
    p.add_argument("--experiment-name", type=str, default=None)
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

# ---------------- Utils: seeding / checkpoint IO -----------------
def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def save_checkpoint(out_dir: Path, tag: str, model, opt, sched, epoch, step, args, keep_last: int = 3):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scheduler": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "global_step": step,
        "args": vars(args),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    path = out_dir / f"ckpt_{tag}.pt"
    torch.save(ckpt, path)

    # encoder / decoder separately
    enc_path = out_dir / f"encoder_{tag}.pt"
    dec_path = out_dir / f"decoder_{tag}.pt"
    torch.save(model.encoder.state_dict(), enc_path)
    torch.save(model.decoder.state_dict(), dec_path)

    # rolling keep_last
    if tag.startswith("step_") and keep_last > 0:
        step_ckpts = sorted(out_dir.glob("ckpt_step_*.pt"))
        excess = len(step_ckpts) - keep_last
        for p in step_ckpts[:max(0, excess)]:
            stem = p.stem.replace("ckpt_", "")
            for ext in [".pt"]:
                for prefix in ["ckpt_", "encoder_", "decoder_"]:
                    q = p.parent / f"{prefix}{stem}{ext}"
                    if q.exists():
                        q.unlink(missing_ok=True)

    return path, enc_path, dec_path

def load_checkpoint(path: str, model, opt=None, sched=None, map_location="auto"):
    import torch as _torch

    ckpt = _torch.load(path, map_location=map_location, weights_only=False)

    # --- 1) filtrujemy niepasujące klucze z innych wersji (proj_skip.*) ---
    state = ckpt["state_dict"]
    state = {k: v for k, v in state.items() if not k.endswith("proj_skip.weight")}

    # --- 2) ładujemy luźno i NIE wołamy drugi raz strict=True ---
    msg = model.load_state_dict(state, strict=False)
    print("[load] missing keys:", msg.missing_keys)
    print("[load] unexpected keys:", msg.unexpected_keys)

    # --- 3) opt/sched – próbujemy, ale nie blokujemy się gdy kształty nie pasują ---
    if opt is not None and ckpt.get("optimizer") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[load] skip optimizer state: {e}")

    if sched is not None and ckpt.get("scheduler") is not None:
        try:
            sched.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[load] skip scheduler state: {e}")

    # --- 4) RNG – opcjonalne; bezpiecznie rzutujemy albo pomijamy ---
    rng = ckpt.get("rng_state")
    if rng:
        try:
            t = rng.get("torch")
            if t is not None:
                if not isinstance(t, _torch.ByteTensor):
                    t = _torch.tensor(t, dtype=_torch.uint8, device="cpu")
                _torch.set_rng_state(t)

            if _torch.cuda.is_available():
                cuda_states = rng.get("cuda")
                if isinstance(cuda_states, list):
                    cuda_states = [
                        (_torch.tensor(cs, dtype=_torch.uint8, device="cuda")
                         if not isinstance(cs, _torch.cuda.ByteTensor) else cs)
                        for cs in cuda_states
                    ]
                    _torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            print(f"[load] RNG restore skipped: {e}")

    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))



# ---------------- Logging (optional) ----------------
class Logger:
    def __init__(self, which="none", project="wikiart-ae", exp_name=None, out_dir: Path | None = None):
        self.which = which
        self.run = None
        self.out_dir = out_dir
        if which == "comet":
            try:
                from comet_ml import Experiment
                self.run = Experiment(project_name=project, auto_output_logging="simple", log_code=False)
                if exp_name: self.run.set_name(exp_name)
            except Exception as e:
                print(f"[logger] Comet init failed: {e}; falling back to none")
                self.which, self.run = "none", None
        elif which == "wandb":
            try:
                import wandb
                self.run = wandb.init(project=project, name=exp_name, dir=str(out_dir) if out_dir else None)
            except Exception as e:
                print(f"[logger] WandB init failed: {e}; falling back to none")
                self.which, self.run = "none", None

    def log(self, metrics: dict, step: int | None = None):
        if self.which == "comet" and self.run:
            self.run.log_metrics(metrics, step=step)
        elif self.which == "wandb" and self.run:
            import wandb
            self.run.log(metrics, step=step)

    def log_image_grid(self, path: Path, name: str, step: int | None = None):
        if not path.exists(): return
        if self.which == "comet" and self.run:
            self.run.log_image(str(path), name=name, step=step)
        elif self.which == "wandb" and self.run:
            import wandb
            self.run.log({name: wandb.Image(str(path))}, step=step)

    def log_artifact(self, path: Path, name: str):
        if self.which == "comet" and self.run:
            try: self.run.log_asset(str(path), file_name=name)
            except Exception: pass
        elif self.which == "wandb" and self.run:
            import wandb
            art = wandb.Artifact(name=name, type="checkpoint")
            art.add_file(str(path))
            self.run.log_artifact(art)

    def finish(self):
        try:
            if self.which == "comet" and self.run: self.run.end()
            elif self.which == "wandb" and self.run:
                import wandb; wandb.finish()
        except Exception:
            pass

# ---------------- Main ----------------
def main():
    args = parse_args()
    seed_all(args.seed)
    
    out_dir = Path(args.out)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    
    logger = Logger(args.logger, project=args.project, exp_name=args.experiment_name, out_dir=out_dir)
    if logger.which != "none":
        logger.log({"cfg/size": args.size, "cfg/os": args.os, "cfg/lr": args.lr}, step=0)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    # fixed eval batch for consistent grids
    try:
        fixed_eval = next(iter(val_ld)).to(device)
    except StopIteration:
        fixed_eval = None


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

    best_psnr = -1.0
    start_epoch, global_step = 0, 0
    
    # --------- RESUME ----------
    if args.resume and os.path.isfile(args.resume):
        print(f"[resume] Loading from: {args.resume}")
        start_epoch, global_step = load_checkpoint(args.resume, model, opt, sched, map_location=device)
        print(f"[resume] epoch={start_epoch}, step={global_step}")

    bad = [n for n,p in model.named_parameters() if p.device.type != device.type]
    if bad:
        print("[warn] params on wrong device:", bad[:5], "…")


    model.to(device)
    
    for m in model.modules():
        if hasattr(m, "proj_skip"):
            m._optimizer_ref = opt
    
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    perc = VGGPerceptual().to(device)

    try:
        for ep in range(start_epoch, args.epochs):
            model.train()
            for imgs in train_ld:
                imgs = imgs.to(device)
                recon, _ = model(imgs)
                loss = l1(recon, imgs) + 0.1 * perc(recon, imgs) + 0.01 * total_variation(recon)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                # logs
                if global_step % 50 == 0:
                    print(f"[train] epoch {ep} step {global_step} | loss={loss.item():.4f}")
                    logger.log({"train/loss": float(loss.item()),
                                "lr": opt.param_groups[0]["lr"]}, step=global_step)

                # grids
                if args.save_every and global_step % args.save_every == 0:
                    with torch.no_grad():
                        grid = torch.cat([(imgs[:8]+1)/2, (recon[:8]+1)/2], dim=0)
                        gpath = out_dir/"samples"/f"train_step_{global_step:06d}.png"
                        save_image(grid, gpath, nrow=8)
                        logger.log_image_grid(gpath, "train_grid", step=global_step)

                        if fixed_eval is not None:
                            rec_eval, _ = model(fixed_eval)
                            grid_eval = torch.cat([(fixed_eval[:8]+1)/2, (rec_eval[:8]+1)/2], dim=0)
                            vpath = out_dir/"samples"/f"val_fixed_step_{global_step:06d}.png"
                            save_image(grid_eval, vpath, nrow=8)
                            logger.log_image_grid(vpath, "val_fixed_grid", step=global_step)


                # step checkpoint
                if args.ckpt_every and global_step > 0 and global_step % args.ckpt_every == 0:
                    tag = f"step_{global_step:06d}"
                    ckp, encp, decp = save_checkpoint(out_dir/"checkpoints", tag, model, opt, sched,
                                                      epoch=ep, step=global_step, args=args, keep_last=args.keep_last)
                    logger.log_artifact(Path(ckp), f"ckpt_{tag}")
                    logger.log_artifact(Path(encp), f"encoder_{tag}")
                    logger.log_artifact(Path(decp), f"decoder_{tag}")

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
                    ps = 10.0 * math.log10((2.0 ** 2) / max(mse, 1e-12))
                    total_psnr += float(ps) * imgs.size(0)
                    if HAS_TM:
                        total_ssim += float(ssim((recon+1)/2, (imgs+1)/2, data_range=1.0)) * imgs.size(0)
                    n += imgs.size(0)

                l1_val = total_l1 / n
                mse_val = total_mse / n
                psnr_val = total_psnr / n
                ssim_val = (total_ssim / n) if HAS_TM else float('nan')

                print(f"[val] epoch {ep}: L1={l1_val:.4f} MSE={mse_val:.6f} PSNR={psnr_val:.2f} SSIM={ssim_val:.4f}")
                logger.log({"val/L1": l1_val, "val/MSE": mse_val, "val/PSNR": psnr_val, "val/SSIM": ssim_val}, step=global_step)

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
                    logger.log_artifact(best_path, "ae_best")
                    
                    torch.save(model.encoder.state_dict(), out_dir / "encoder_best.pt")
                    torch.save(model.decoder.state_dict(), out_dir / "decoder_best.pt")
                    logger.log_artifact(out_dir / "encoder_best.pt", "encoder_best")
                    logger.log_artifact(out_dir / "decoder_best.pt", "decoder_best")

                if args.ckpt_epoch:
                    tag = f"epoch_{ep:03d}"
                    ckp, encp, decp = save_checkpoint(out_dir/"checkpoints", tag, model, opt, sched,
                        epoch=ep, step=global_step, args=args, keep_last=0)
                    logger.log_artifact(Path(ckp), f"ckpt_{tag}")
                    logger.log_artifact(Path(encp), f"encoder_{tag}")
                    logger.log_artifact(Path(decp), f"decoder_{tag}")

            sched.step()

        # save last
        last_path = out_dir / "ae_last.pt"
        torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": args.epochs}, last_path)
        print(f"Saved last checkpoint to: {last_path}")
        logger.log_artifact(last_path, "ae_last")
        
        torch.save(model.encoder.state_dict(), out_dir / "encoder_last.pt")
        torch.save(model.decoder.state_dict(), out_dir / "decoder_last.pt")
        logger.log_artifact(out_dir / "encoder_last.pt", "encoder_last")
        logger.log_artifact(out_dir / "decoder_last.pt", "decoder_last")
    
    except KeyboardInterrupt:
        # safe save during epoch
        print("\n[train] KeyboardInterrupt — saving interrupted checkpoint...")
        tag = "interrupted"
        ckp, encp, decp = save_checkpoint(out_dir/"checkpoints", tag, model, opt, sched,
                                          epoch=ep if 'ep' in locals() else -1,
                                          step=global_step, args=args, keep_last=0)
        print(f"[train] Saved: {ckp}")
        logger.log_artifact(Path(ckp), f"ckpt_{tag}")
        logger.log_artifact(Path(encp), f"encoder_{tag}")
        logger.log_artifact(Path(decp), f"decoder_{tag}")

    finally:
        logger.finish()


if __name__ == "__main__":
    main()
