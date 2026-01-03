"""database.py

Utilities for downloading, splitting and accessing the WikiArt datasets for
both tasks in the project:
- Inpainting (default dataset: "Artificio/WikiArt_Full")
- Super-resolution (default dataset: "huggan/wikiart")

Key features
------------
- Programmatic download from the Hugging Face Hub via `datasets`.
- Reproducible 3-way split: train/val/test with a single seed and optional
  stratification.
- Simple getter functions: `get_train()`, `get_val()`, `get_test()`.
- Save/Load prepared splits to/from disk.
- (Optional) PyTorch adapter; and an SR helper that produces (LR, HR) pairs
  on-the-fly by downscaling the original image.

Requirements
------------
- datasets >= 2.14.0
- pillow
- (optional) torch, torchvision – for the Torch adapter

Example
-------
>>> # INPAINTING
>>> db = ArtDatabase(task="inpainting")  # default Artificio/WikiArt_Full
>>> db.download()
>>> db.make_split(val_size=0.1, test_size=0.1, seed=123, stratify_by="style")
>>> db.save_splits("./splits/wikiart_full_inpaint")
>>> train = db.get_train(); val = db.get_val(); test = db.get_test()

>>> # SUPER-RESOLUTION
>>> sr = ArtDatabase(task="superres")  # default huggan/wikiart
>>> sr.download()
>>> sr.make_split(val_size=0.05, test_size=0.1, seed=123, stratify_by="style")
>>> sr_pairs = sr.as_superres_pairs(split="val", downscale=4)  # (LR, HR)

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Tuple
import warnings

try:
    from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "The 'datasets' package is required. Install with: pip install datasets"
    ) from e

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Pillow is required. Install with: pip install pillow"
    ) from e


_DEFAULTS = {
    "inpainting": "Artificio/WikiArt_Full",
    "superres": "huggan/wikiart",
}


@dataclass
class ArtDatabase:
    """Manager for a Hugging Face image dataset used in this project.

    Parameters
    ----------
    task : Optional[str]
        One of {"inpainting", "superres"}. Determines the default dataset_id if
        not provided.
    dataset_id : Optional[str]
        HF Hub dataset identifier. If None, uses a default based on `task`.
    cache_dir : Optional[str]
        Where to cache the raw HF data (mirrors `datasets.load_dataset`).
    local_repo_dir : Optional[str]
        If you've already `git lfs clone`d the dataset, point here.
    """

    task: Optional[str] = None
    dataset_id: Optional[str] = None
    cache_dir: Optional[str] = None
    local_repo_dir: Optional[str] = None

    # internal state
    _raw: Optional[Dataset] = None
    _splits: Optional[DatasetDict] = None

    # -----------------------------
    # Download / prepare
    # -----------------------------
    def _resolve_dataset_id(self) -> str:
        if self.dataset_id:
            return self.dataset_id
        if self.task and self.task in _DEFAULTS:
            return _DEFAULTS[self.task]
        # Fallback to a sensible default
        return _DEFAULTS["inpainting"]

    def download(self, streaming: bool = False) -> Dataset:
        """ Download the dataset from the HF Hub.
       """
        ds_id = self._resolve_dataset_id()
        self._raw = load_dataset(
            path=ds_id,
            split="train",
            cache_dir=self.cache_dir,
            streaming=streaming,
            data_dir=self.local_repo_dir,
        )
        return self._raw

    def make_split(
        self,
        val_size: float = 0,
        test_size: float = 0.3,
        seed: int = 42,
        stratify_by: Optional[str] = "style",
        shuffle: bool = True,
    ) -> DatasetDict:
        """Create a reproducible 3-way train/val/test split.

        If `val_size` is 0, produces a 2-way train/test split.
        """
        if self._raw is None:
            raise RuntimeError("Call download() or load_splits() before make_split().")
        if not (0 <= val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
            raise ValueError("val_size and test_size must be in (0,1) and sum < 1.")

        # Ensure stratify column is of type ClassLabel if requested (HF requires this)
        use_stratify = False
        if stratify_by is not None and stratify_by in self._raw.column_names:
            try:
                from datasets.features import ClassLabel
                feat = self._raw.features.get(stratify_by)
                if not isinstance(feat, ClassLabel):
                    self._raw = self._raw.class_encode_column(stratify_by)
                # Check class counts; HF requires at least 2 samples per class
                try:
                    ser = self._raw.to_pandas()[stratify_by]
                    min_count = int(ser.value_counts().min())
                    if min_count < 2:
                        warnings.warn(
                            f"Stratification disabled: column '{stratify_by}' contains classes with <2 samples (min={min_count}).",
                            RuntimeWarning,
                        )
                        use_stratify = False
                    else:
                        use_stratify = isinstance(self._raw.features.get(stratify_by), ClassLabel)
                except Exception:
                    # If counting fails for any reason, fall back to non-stratified
                    use_stratify = False
            except Exception:
                use_stratify = False

        # First carve out TEST
        kwargs: Dict[str, Any] = {
            "test_size": test_size,
            "seed": seed,
            "shuffle": shuffle,
        }
        if use_stratify:
            kwargs["stratify_by_column"] = stratify_by
        base = self._raw.train_test_split(**kwargs)
        test = base["test"]
        remaining = base["train"]

        if val_size > 0:
            rel_val = val_size / (1.0 - test_size)
            kwargs2 = {
                "test_size": rel_val,
                "seed": seed,
                "shuffle": shuffle,
            }
            if use_stratify and stratify_by in remaining.column_names:
                kwargs2["stratify_by_column"] = stratify_by
            tmp = remaining.train_test_split(**kwargs2)
            train, val = tmp["train"], tmp["test"]
            self._splits = DatasetDict({"train": train, "val": val, "test": test})
        else:
            self._splits = DatasetDict({"train": remaining, "test": test})
        return self._splits

    # -----------------------------
    # Persistence
    # -----------------------------
    def save_splits(self, path: str) -> None:
        if self._splits is None:
            raise RuntimeError("No splits to save. Call make_split() first.")
        self._splits.save_to_disk(path)

    def load_splits(self, path: str) -> DatasetDict:
        ds = load_from_disk(path)
        if isinstance(ds, Dataset):
            ds = DatasetDict({"train": ds})
        self._splits = ds
        return self._splits

    # -----------------------------
    # Getters
    # -----------------------------
    def get_split(self, name: str) -> Dataset:
        if self._splits is None:
            raise RuntimeError("No splits available. Call make_split() or load_splits().")
        if name not in self._splits:
            raise KeyError(f"Split '{name}' not present. Available: {list(self._splits.keys())}")
        return self._splits[name]

    def get_train(self) -> Dataset:
        return self.get_split("train")

    def get_val(self) -> Dataset:
        return self.get_split("val")

    def get_test(self) -> Dataset:
        return self.get_split("test")

    # -----------------------------
    # Torch adapter (optional)
    # -----------------------------
    def as_torch(self, split: str = "train", transform: Optional[Callable] = None):
        """Return a lightweight PyTorch-style dataset wrapper.

        Yields `(PIL.Image, labels_dict)` with keys among {style, genre, artist}.
        """
        try:
            import torch  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyTorch is required for as_torch(). pip install torch") from e

        ds = self.get_split(split)
        return _TorchWrapper(ds, transform)

    # -----------------------------
    # Super-resolution helper
    # -----------------------------
    def as_superres_pairs(
        self,
        split: str = "train",
        downscale: int = 4,
        resize_interpolation: str = "bicubic",
    ):
        """Return a dataset wrapper that yields `(LR, HR)` PIL image pairs.

        LR is produced on the fly by downscaling HR by `downscale` and then
        (optionally) leaving it at that size for a typical SR pipeline.
        """
        ds = self.get_split(split)

        pil_interp = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[resize_interpolation]

        return _SRWrapper(ds, downscale, pil_interp)


class _TorchWrapper:
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform
        self.label_cols = [c for c in ["style", "genre", "artist"] if c in base.column_names]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        row = self.base[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        labels = {k: row.get(k) for k in self.label_cols}
        return img, labels


class _SRWrapper:
    def __init__(self, base, downscale, pil_interp):
        self.base = base
        self.downscale = downscale
        self.pil_interp = pil_interp

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx) -> Tuple[Image.Image, Image.Image]:
        row = self.base[idx]
        hr = row["image"]
        if not isinstance(hr, Image.Image):
            hr = Image.fromarray(hr)
        w, h = hr.size
        lr = hr.resize((max(1, w // self.downscale), max(1, h // self.downscale)), resample=self.pil_interp)
        return lr, hr


__all__ = ["ArtDatabase"]
