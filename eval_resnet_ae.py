from __future__ import annotations
import argparse, math, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image

from database import ArtDatabase
from autoencoders.resnet_autoencoder import ResNetAutoEncoder

try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    HAS_TM = True
except Exception:
    HAS_TM = False


def psnr_from_mse(mse: float, peak_to_peak: float = 2.0) -> float:
    return 10.0 * math.log10((peak_to_peak ** 2) / max(mse, 1e-12))


def parse_args():
    p = argparse.ArgumentParser("Eval on TEST split")
    p.add_argument("--splits", required=True, help="Folder ze splitami (ten co zrobiłeś prepare_*).")
    p.add_argument("--cache-dir", default="./data/hf_cache")
    p.add_argument("--out", default="./runs/eval")
    p.add_argument("--ckpt", required=True, help="Ścieżka do checkpointu (encoder_*.pt / ae_*.pt / ckpt_*.pt)")
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18","resnet34","resnet50","resnet101","resnet152"])
    p.add_argument("--os", type=int, default=16, choices=[8,16,32])
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=(0 if os.name=="nt" else 4))
    p.add_argument("--mode", choices=["features","recon"], default=None,
                   help="features: wyciąganie wektorów; recon: rekonstrukcja + metryki."
                        " Jeśli nie podasz, zostanie wybrane automatycznie na podstawie nazwy ckpt.")
    p.add_argument("--pool", choices=["gap","none"], default="gap",
                   help="Jak tworzyć wektor cech w trybie features: GAP=global avg pool do 1x1, none=pełna mapa.")
    p.add_argument("--samples", type=int, default=16,
                   help="Ile przykładów zapisać jako obrazki w trybie recon.")
    return p.parse_args()


def load_weights(model: ResNetAutoEncoder, ckpt_path: str) -> str:
    path = Path(ckpt_path)
    sd = torch.load(str(path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        state = sd["state_dict"]
    else:
        state = sd

    keys = list(state.keys())
    is_encoder_only = all(k.startswith("encoder.") for k in keys) or \
                      (any(k.startswith("encoder.") for k in keys) and not any(k.startswith("decoder.") for k in keys))

    if is_encoder_only:
        msg = model.encoder.load_state_dict({k.replace("encoder.", "") if k.startswith("encoder.") else k: v
                                             for k,v in state.items()}, strict=False)
        print("[load][encoder-only] missing:", msg.missing_keys, "unexpected:", msg.unexpected_keys)
        return "encoder"
    else:
        msg = model.load_state_dict(state, strict=False)
        print("[load][full-ae] missing:", msg.missing_keys, "unexpected:", msg.unexpected_keys)
        return "ae"


def main():
    args = parse_args()
    out_dir = Path(args.out); (out_dir/"samples").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    db = ArtDatabase(task="inpainting", cache_dir=args.cache_dir)
    db.load_splits(args.splits)

    test_hf = db.get_test()
    print("[eval] TEST samples:", len(test_hf))

    transform = T.Compose([
        T.Resize(args.size),
        T.CenterCrop(args.size),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    test_ds = db.as_torch(split="test", transform=transform)

    def collate(batch):
        return torch.stack([b[0] for b in batch], dim=0)

    pin = torch.cuda.is_available()
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=pin, persistent_workers=False,
                         collate_fn=collate)

    model = ResNetAutoEncoder(
        name=args.backbone, pretrained=False, output_stride=args.os,
        use_skips=True, out_channels=3, base_width=256, num_ups=None, final_act="tanh"
    ).to(device)

    kind = load_weights(model, args.ckpt)

    mode = args.mode or ("features" if kind == "encoder" else "recon")
    print(f"[eval] mode={mode}")

    model.eval()
    with torch.no_grad():
        if mode == "features":
            ids, feats = [], []
            for imgs in test_ld:
                imgs = imgs.to(device)
                enc_out = model.encoder(imgs)
                top = enc_out[0] if isinstance(enc_out, (list, tuple)) else enc_out  # C5
                if args.pool == "gap":
                    v = torch.nn.functional.adaptive_avg_pool2d(top, output_size=1).squeeze(-1).squeeze(-1)  # [B,C]
                else:
                    v = top.flatten(1)  # [B, C*H*W]

                feats.append(v.cpu().numpy())
                b = v.size(0)
                start = len(ids)
                ids.extend([f"test_{start+i}" for i in range(b)])

            feats = np.concatenate(feats, axis=0)
            out_npz = out_dir / "test_features.npz"
            np.savez_compressed(out_npz, ids=np.array(ids, dtype=object), features=feats)
            print(f"[eval][features] saved: {out_npz}  shape={feats.shape}")

        else:  # recon
            (out_dir / "samples").mkdir(parents=True, exist_ok=True)

            tot_l1 = tot_mse = tot_psnr = tot_ssim = n = 0
            save_all = (args.samples is None) or (args.samples <= 0)
            saved = 0
            img_index = 0

            for imgs in test_ld:
                imgs = imgs.to(device)
                recon, _ = model(imgs)

                l1 = torch.mean(torch.abs(recon - imgs)).item()
                mse = torch.mean((recon - imgs) ** 2).item()
                ps  = psnr_from_mse(mse)
                if HAS_TM:
                    s = float(ssim((recon+1)/2, (imgs+1)/2, data_range=1.0))
                else:
                    s = float("nan")

                bs = imgs.size(0)
                tot_l1  += l1 * bs
                tot_mse += mse * bs
                tot_psnr+= ps * bs
                if not math.isnan(s):
                    tot_ssim+= s * bs
                n += bs

                if save_all:
                    k = bs
                else:
                    if saved >= args.samples:
                        break
                    k = min(bs, args.samples - saved)

                if k > 0:
                    grid = torch.cat([(imgs[:k] + 1) / 2, (recon[:k] + 1) / 2], dim=0)
                    save_image(grid, out_dir / "samples" / f"test_{img_index:06d}.png", nrow=k)
                    saved += k
                    img_index += k

                if (not save_all) and (saved >= args.samples):
                    break

            l1_val  = tot_l1 / n
            mse_val = tot_mse / n
            psnr_val= tot_psnr / n
            ssim_val= (tot_ssim / n) if HAS_TM else float('nan')
            print(f"[eval][recon] L1={l1_val:.4f}  MSE={mse_val:.6f}  PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}")

            with open(out_dir/"metrics_test.txt", "w", encoding="utf-8") as f:
                f.write(f"L1={l1_val:.6f}\nMSE={mse_val:.8f}\nPSNR={psnr_val:.3f}\nSSIM={ssim_val:.5f}\n")
            print(f"[eval][recon] metrics saved to {out_dir/'metrics_test.txt'}")

if __name__ == "__main__":
    main()
