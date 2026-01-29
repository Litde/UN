"""
Inpainting (Artificio/WikiArt_Full):
python demo_database.py --task inpainting --splits ./splits/wikiart_full_inpaint --val-size 0.1 --test-size 0.1 --seed 123

Super-resolution (huggan/wikiart):
python demo_database.py --task superres --splits ./splits/huggan_sr --val-size 0.05 --test-size 0.1 --seed 123 --downscale 4
"""
from __future__ import annotations

import argparse
import os

from database import ArtDatabase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["inpainting", "superres"], required=True,
                   help="Which task preset to use (controls default dataset_id).")
    p.add_argument("--splits", type=str, required=True,
                   help="Directory to load/save dataset splits.")
    p.add_argument("--dataset-id", type=str, default=None,
                   help="Optional explicit HF dataset id (overrides task default).")
    p.add_argument("--val-size", type=float, default=0.1,
                   help="Validation fraction (0 disables val split).")
    p.add_argument("--test-size", type=float, default=0.1,
                   help="Test fraction.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    p.add_argument("--stratify-by", type=str, default="style",
                   help="Optional column name to stratify by (e.g. style/genre). Use 'none' to disable.")
    p.add_argument("--downscale", type=int, default=4,
                   help="For superres pairs: downscale factor.")
    return p.parse_args()


def main():
    args = parse_args()
    strat = None if args.stratify_by.lower() == "none" else args.stratify_by

    db = ArtDatabase(task=args.task, dataset_id=args.dataset_id, cache_dir="./data/hf_cache")

    if os.path.isdir(args.splits):
        print(f"[demo] Loading existing splits from: {args.splits}")
        db.load_splits(args.splits)
    else:
        print("[demo] No splits found; downloading and creating new splits...")
        db.download()
        db.make_split(val_size=args.val_size, test_size=args.test_size,
                      seed=args.seed, stratify_by=strat)
        db.save_splits(args.splits)
        print(f"[demo] Saved splits to: {args.splits}")

    # Basic stats
    names = [k for k in ["train", "val", "test"] if k in getattr(db, "_splits").keys()]
    for n in names:
        print(f"[demo] {n}: {len(db.get_split(n))} samples")

    # Quick peek for inpainting
    if args.task == "inpainting":
        ds = db.get_train()
        row = ds[0]
        print(f"[demo] Sample keys: {list(row.keys())}")
        print(f"[demo] Labels present: {[k for k in ['style','genre','artist'] if k in row]}")

    # Quick SR preview
    if args.task == "superres":
        pairs = db.as_superres_pairs(split="val" if "val" in names else "train",
                                     downscale=args.downscale)
        lr, hr = pairs[0]
        print(f"[demo] SR pair sizes: LR={lr.size}, HR={hr.size}")


if __name__ == "__main__":
    main()
