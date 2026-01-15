"""prepare_wikiart_full.py

Download the full **Artificio/WikiArt_Full** dataset and create a
reproducible 70/15/15 (train/val/test) split on disk using `database.ArtDatabase`.

Usage
-----
# from the project root
python scripts/prepare_wikiart_full.py \
  --out ./splits/wikiart_full_70_15_15 \
  --cache-dir ./data/hf_cache \
  --seed 123 \
  --stratify-by style

Notes
-----
- If the output directory already exists, the script will **load** the splits
  instead of recreating them.
- Stratification falls back to non‑stratified automatically if the column has
  too few samples in some classes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from database import ArtDatabase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare 70/15/15 splits for Artificio/WikiArt_Full")
    p.add_argument("--out", type=str, required=True,
                   help="Where to save the DatasetDict splits (directory).")
    p.add_argument("--cache-dir", type=str, default="./data/hf_cache",
                   help="Hugging Face cache directory for raw data.")
    p.add_argument("--seed", type=int, default=123, help="Random seed for splitting.")
    p.add_argument("--stratify-by", type=str, default="style",
                   help="Optional column to stratify by (e.g. style/genre/artist). Use 'none' to disable.")
    p.add_argument("--dataset-id", type=str, default=None,
                   help="Optional dataset id (defaults to Artificio/WikiArt_Full).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    strat = None if (args.stratify_by is None or args.stratify_by.lower() == "none") else args.stratify_by

    db = ArtDatabase(task="inpainting", dataset_id=args.dataset_id, cache_dir=args.cache_dir)

    if out_dir.exists():
        print(f"[prepare] Output exists, loading splits from: {out_dir}")
        db.load_splits(str(out_dir))
    else:
        print("[prepare] Downloading dataset (this may take a while on first run)...")
        db.download()
        print("[prepare] Creating 70/15/15 split (train/val/test)...")
        db.make_split(val_size=0.15, test_size=0.15, seed=args.seed, stratify_by=strat)
        print(f"[prepare] Saving splits to: {out_dir}")
        db.save_splits(str(out_dir))

    # Report basic stats
    names = [k for k in ["train", "val", "test"] if k in getattr(db, "_splits").keys()]
    for n in names:
        print(f"[prepare] {n:>5}: {len(db.get_split(n))} samples")

    # Quick peek
    row = db.get_train()[0]
    print("[prepare] sample keys:", list(row.keys()))
    print("[prepare] label cols present:", [k for k in ["style", "genre", "artist"] if k in row])


if __name__ == "__main__":
    main()
