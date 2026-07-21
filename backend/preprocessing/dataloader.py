"""
DEFECTRA - Multi-Task Learning Dataloader
==========================================
Dataset   : Carinthia Semiconductor Dataset
Tasks     : Binary Segmentation + Defect Classification (6 classes)
Encoder   : EfficientNet-B2
Framework : PyTorch 2.x + Albumentations
Python    : 3.11

Requirements (install via pip):
    torch>=2.0.0
    torchvision>=0.15.0
    albumentations>=1.3.0
    Pillow>=9.0.0
    numpy>=1.24.0
    pandas>=2.0.0
    segmentation-models-pytorch>=0.3.3
    opencv-python-headless>=4.7.0

Run sanity check:
    python preprocessing/dataloader.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# Resolve backend/ directory so "from config import ..." always works
# regardless of the working directory the script is invoked from.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
from config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    CSV_PATH,
    SPLIT_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    PIN_MEMORY,
    USE_AUGMENTATION,
    TRAIN_AUG,
    VAL_AUG,
    TEST_AUG,
    SHUFFLE_TRAIN,
    MEAN,
    STD,
)
# ---------------------------------------------------------------------------
# Pillow compatibility: Image.NEAREST was removed in Pillow 10.
# Use Image.Resampling.NEAREST when available, fall back to Image.NEAREST.
# ---------------------------------------------------------------------------
try:
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:
    _NEAREST = Image.NEAREST  # type: ignore[attr-defined]  # Pillow < 10

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = MEAN
IMAGENET_STD = STD

# ---------------------------------------------------------------------------
# Internal helpers for albumentations v1 / v2 compatibility
# ---------------------------------------------------------------------------

def _blur_limit(value):
    """
    Normalise blur_limit so it works with both albumentations v1 and v2.

    albumentations >= 2.0 requires blur_limit to be a tuple of two odd
    positive integers, e.g. (3, 7).  v1 accepts a plain int (max value).
    This helper converts a plain int from config into a safe tuple.

    Parameters
    ----------
    value : int or tuple
        Value read from TRAIN_AUG["blur_limit"] in config.py.

    Returns
    -------
    tuple[int, int]
        A (min, max) pair where both values are odd and min <= max.
    """
    if isinstance(value, (list, tuple)):
        lo, hi = int(value[0]), int(value[1])
    else:
        hi = int(value)
        lo = 3  # sensible minimum kernel size
    # Ensure both limits are odd (required by GaussianBlur)
    if lo % 2 == 0:
        lo += 1
    if hi % 2 == 0:
        hi += 1
    if lo > hi:
        lo = hi
    return (lo, hi)


def _rotate_limit(value):
    """
    Normalise rotate_limit for albumentations v1 / v2 compatibility.

    albumentations >= 2.0 prefers a tuple (-limit, limit).  v1 accepts a
    plain int.  This helper converts a plain int from config into a tuple.

    Parameters
    ----------
    value : int or tuple
        Value read from TRAIN_AUG["rotate_limit"] in config.py.

    Returns
    -------
    tuple[int, int]
        A (-limit, +limit) pair.
    """
    if isinstance(value, (list, tuple)):
        return (int(value[0]), int(value[1]))
    limit = int(value)
    return (-limit, limit)


# ---------------------------------------------------------------------------
# Augmentation pipeline builders
# ---------------------------------------------------------------------------

def _build_train_transform():

    aug = []

    aug.append(A.Resize(IMAGE_SIZE, IMAGE_SIZE))

    if TRAIN_AUG["horizontal_flip_prob"] > 0:
        aug.append(
            A.HorizontalFlip(
                p=TRAIN_AUG["horizontal_flip_prob"]
            )
        )

    if TRAIN_AUG["vertical_flip_prob"] > 0:
        aug.append(
            A.VerticalFlip(
                p=TRAIN_AUG["vertical_flip_prob"]
            )
        )

    if TRAIN_AUG["random_rotate90_prob"] > 0:
        aug.append(
            A.RandomRotate90(
                p=TRAIN_AUG["random_rotate90_prob"]
            )
        )

    aug.append(
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
            )
    )

    aug.append(ToTensorV2())

    return A.Compose(aug)

def _build_val_transform() -> A.Compose:
    """
    Build the validation pipeline (Resize + Normalize only — no geometric
    augmentation).  VAL_AUG is reserved for future configurable parameters
    (e.g. test-time interpolation method) and is imported from config.py.
    """
    _ = VAL_AUG  # imported from config; reserved for future parameters
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
            ),
        ToTensorV2(),
    ])


def _build_test_transform() -> A.Compose:
    """
    Build the test pipeline (Resize + Normalize only — no geometric
    augmentation).  TEST_AUG is reserved for future configurable parameters
    and is imported from config.py.
    """
    _ = TEST_AUG  # imported from config; reserved for future parameters
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
            ),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# CSV helper: build O(1) lookup dictionaries (parsed once at startup)
# ---------------------------------------------------------------------------

def _build_lookup_dicts(csv_path: str):
    """
    Parse carinthia-s.csv once and return three O(1) lookup dictionaries.

    Labels in the CSV are 1-based (1-6) and are converted to 0-based
    (0-5) here so they are ready for CrossEntropyLoss without any further
    transformation downstream.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to carinthia-s.csv.

    Returns
    -------
    filename_to_label : dict
        Maps full filename  (e.g. "0001.png") -> zero-indexed int label.
    stem_to_label : dict
        Maps filename stem  (e.g. "0001")     -> zero-indexed int label.
    stem_to_mask : dict
        Maps filename stem  (e.g. "0001")     -> mask path string from CSV.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"[DEFECTRA] CSV not found: '{csv_path}'\n"
            "Check CSV_PATH in config.py."
        )

    try:
        df = pd.read_csv(csv_path)
        
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path, sep=";")
    except Exception as e:
        raise RuntimeError(f"[DEFECTRA] Unable to read CSV: {e}")

    required_cols = {"image_path", "mask_path", "filename", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[DEFECTRA] CSV is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Validate label range (1-NUM_CLASSES) before conversion
    valid_mask = df["label"].between(1, NUM_CLASSES)
    if not valid_mask.all():
        bad_vals = df.loc[~valid_mask, "label"].unique().tolist()
        raise ValueError(
            f"[DEFECTRA] CSV contains unexpected label values: {bad_vals}. "
            f"Expected integer labels in [1, {NUM_CLASSES}]."
        )

    # Convert labels 1-6 -> 0-5 (CrossEntropyLoss convention)
    df["label_idx"] = df["label"].astype(int) - 1

    filename_to_label = {}
    stem_to_label = {}
    stem_to_mask = {}

    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        stem  = Path(fname).stem
        label = int(row["label_idx"])
        mask  = str(row["mask_path"]).strip()

        filename_to_label[fname] = label
        stem_to_label[stem]      = label
        stem_to_mask[stem]       = mask

    return filename_to_label, stem_to_label, stem_to_mask


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DefectraDataset(Dataset):
    """
    PyTorch Dataset for the DEFECTRA multi-task learning pipeline.

    Each call to __getitem__ returns a three-tuple:

        image_tensor : torch.float32  shape (3, H, W)  -- ImageNet-normalised
        mask_tensor  : torch.float32  shape (1, H, W)  -- binary {0.0, 1.0}
        label_tensor : torch.long     shape ()          -- zero-indexed (0-5)

    Invalid samples (missing image / mask / label entry) are skipped with a
    printed warning so that training is never interrupted by a single bad file.
    """

    def __init__(
        self,
        split: str,
        filename_to_label: dict,
        stem_to_label: dict,
        stem_to_mask: dict,
        transform: A.Compose,
    ) -> None:
        """
        Parameters
        ----------
        split             : 'train', 'val', or 'test'
        filename_to_label : {filename -> 0-indexed label}  built from CSV
        stem_to_label     : {stem     -> 0-indexed label}  built from CSV
        stem_to_mask      : {stem     -> mask path string} built from CSV
        transform         : Albumentations Compose pipeline
        """
        self.transform         = transform
        self.filename_to_label = filename_to_label
        self.stem_to_label     = stem_to_label
        self.stem_to_mask      = stem_to_mask

        image_dir = Path(SPLIT_DIR) / split / "images"
        mask_dir  = Path(SPLIT_DIR) / split / "masks"

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"[DEFECTRA] Image directory not found: '{image_dir}'\n"
                "Check SPLIT_DIR in config.py."
            )
        if not mask_dir.is_dir():
            raise FileNotFoundError(
                f"[DEFECTRA] Mask directory not found: '{mask_dir}'\n"
                "Check SPLIT_DIR in config.py."
            )

        self.samples = []  # list of (img_path_str, mask_path_str, label_int)
        skipped = 0

        for img_file in sorted(image_dir.iterdir()):
            if img_file.suffix.lower() not in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
            }:
                continue

            fname = img_file.name
            stem  = img_file.stem

            # ------------------------------------------------------------------
            # 1. Resolve classification label (O(1) dict lookup)
            # ------------------------------------------------------------------
            if fname in filename_to_label:
                label = filename_to_label[fname]
            elif stem in stem_to_label:
                label = stem_to_label[stem]
            else:
                print(
                    f"[DEFECTRA] WARNING: No CSV label found for '{fname}' "
                    f"(split='{split}'). Skipping."
                )
                skipped += 1
                continue

            # ------------------------------------------------------------------
            # 2. Resolve mask path
            #    Priority 1 : path recorded in the CSV
            #    Priority 2 : <split>/masks/<same filename>
            #    Priority 3 : <split>/masks/<stem>.<common extension>
            # ------------------------------------------------------------------
            mask_path = None

            if stem in stem_to_mask:
                candidate = Path(stem_to_mask[stem])
                if not candidate.is_absolute():
                    # Relative path in CSV: anchor to project root (two levels
                    # above SPLIT_DIR, i.e. the dataset/ parent directory).
                    candidate = Path(SPLIT_DIR).parent.parent / candidate
                if candidate.is_file():
                    mask_path = candidate

            if mask_path is None:
                # Try the split masks folder with the same name first
                candidate = mask_dir / fname
                if candidate.is_file():
                    mask_path = candidate

            if mask_path is None:
                # Fallback: try common image extensions
                for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                    candidate = mask_dir / (stem + ext)
                    if candidate.is_file():
                        mask_path = candidate
                        break

            if mask_path is None:
                print(
                    f"[DEFECTRA] WARNING: Mask not found for '{fname}' "
                    f"(split='{split}'). Skipping."
                )
                skipped += 1
                continue

            self.samples.append((str(img_file), str(mask_path), label))

        if skipped:
            print(
                f"[DEFECTRA] {skipped} sample(s) skipped in split='{split}' "
                "(missing image / mask / label)."
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[DEFECTRA] No valid samples found for split='{split}'. "
                "Check SPLIT_DIR and CSV_PATH in config.py."
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_path, mask_path, label = self.samples[idx]

        # ------------------------------------------------------------------
        # Load image as RGB uint8 numpy array (H x W x 3).
        # Albumentations expects channels-last uint8 for the image input.
        # ------------------------------------------------------------------
        try:
            image = np.array(
                Image.open(img_path).convert("RGB"),
                dtype=np.uint8,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[DEFECTRA] Cannot load image '{img_path}': {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Load mask as grayscale float32 numpy array (H x W).
        # Binarise: pixels > 127 are foreground (1.0), rest background (0.0).
        # NOTE: do NOT resize here – A.Resize inside self.transform handles
        #       resizing for both image and mask in a single, consistent step.
        # ------------------------------------------------------------------
        try:
            mask_pil = Image.open(mask_path).convert("L")
            mask_np  = np.array(mask_pil, dtype=np.float32)
            mask_np  = (mask_np > 127).astype(np.float32)  # binary {0.0, 1.0}
        except Exception as exc:
            raise RuntimeError(
                f"[DEFECTRA] Cannot load mask '{mask_path}': {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Apply Albumentations pipeline.
        # The 'mask' keyword tells albumentations to apply spatial transforms
        # (Resize, Rotate, Flip ...) to the mask but skip colour transforms
        # (Normalize, BrightnessContrast ...).
        # ToTensorV2 converts:
        #   image : (H, W, 3) uint8  -> (3, H, W) float32
        #   mask  : (H, W)   float32 -> (H, W)   float32  tensor
        # ------------------------------------------------------------------
        augmented    = self.transform(image=image, mask=mask_np)
        image_tensor = augmented["image"]        # (3, H, W) float32
        mask_tensor  = augmented["mask"]         # (H, W)   float32

        # Add the channel dimension required by segmentation losses:
        # (H, W) -> (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.float()

        # Classification label as a scalar long tensor (0-indexed)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, mask_tensor, label_tensor


# ---------------------------------------------------------------------------
# Public factory: build all three DataLoaders in one call
# ---------------------------------------------------------------------------

def get_dataloaders():
    """
    Parse the CSV, build Datasets for every split, and wrap them in
    DataLoaders.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    label_to_idx : dict[int, int]
        Maps original CSV labels (1-6) to zero-indexed labels (0-5).
        Useful for display / inverse-mapping in predict.py.
    """
    # Build lookup dicts once (O(N) CSV scan, O(1) per-sample lookup later)
    filename_to_label, stem_to_label, stem_to_mask = _build_lookup_dicts(CSV_PATH)

    # label_to_idx: original CSV label (1-6) -> model label (0-5)
    label_to_idx = {orig: orig - 1 for orig in range(1, NUM_CLASSES + 1)}

    # Select transforms according to USE_AUGMENTATION flag
    if USE_AUGMENTATION:
        train_transform = _build_train_transform()
    else:
        # Training without augmentation mirrors the validation pipeline
        train_transform = _build_val_transform()

    val_transform  = _build_val_transform()
    test_transform = _build_test_transform()

    # Build datasets
    shared_kwargs = dict(
        filename_to_label=filename_to_label,
        stem_to_label=stem_to_label,
        stem_to_mask=stem_to_mask,
    )

    train_dataset = DefectraDataset(split="train", transform=train_transform, **shared_kwargs)
    val_dataset   = DefectraDataset(split="val",   transform=val_transform,   **shared_kwargs)
    test_dataset  = DefectraDataset(split="test",  transform=test_transform,  **shared_kwargs)

    # Shared DataLoader settings
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    train_loader = DataLoader(train_dataset, shuffle=SHUFFLE_TRAIN, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False,          **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False,          **loader_kwargs)

    return train_loader, val_loader, test_loader, label_to_idx


# ---------------------------------------------------------------------------
# __main__ - sanity check
# Run from the backend/ directory: python preprocessing/dataloader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 53)
    print("  DEFECTRA MULTI-TASK DATALOADER")
    print("=" * 53)

    train_loader, val_loader, test_loader, label_to_idx = get_dataloaders()

    print(f"Training Images        : {len(train_loader.dataset)}")
    print(f"Validation Images      : {len(val_loader.dataset)}")
    print(f"Testing Images         : {len(test_loader.dataset)}")

    # Pull one batch from the training loader for shape verification
    images, masks, labels = next(iter(train_loader))
    print()
    print("Image Min :", images.min().item())
    print("Image Max :", images.max().item())
    print("Mask Unique :", torch.unique(masks))
    print("Label Range :", labels.min().item(), labels.max().item())

    print(f"Image Shape            : {tuple(images.shape)}  | dtype={images.dtype}")
    print(f"Mask Shape             : {tuple(masks.shape)}   | dtype={masks.dtype}")
    print(f"Label Shape            : {tuple(labels.shape)}  | dtype={labels.dtype}")
    print(f"Sample Labels          : {labels.tolist()}")
    print(f"Classification Classes : {NUM_CLASSES}  (0-indexed: 0 to {NUM_CLASSES - 1})")
    print(f"Segmentation Classes   : 2  (background=0, foreground=1)")
    print("=" * 53)
