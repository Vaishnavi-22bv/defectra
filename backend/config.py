# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

import os
import torch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# ==========================================================
# Kaggle Environment
# ==========================================================

if os.path.exists("/kaggle/input"):

    DATASET_DIR = "/kaggle/input/datasets/nanivaishu/defectra-dataset"

    CSV_PATH = os.path.join(
        DATASET_DIR,
        "carinthia-s.csv"
    )

    SPLIT_DIR = os.path.join(
        DATASET_DIR,
        "split_kaggle"
    )

# ==========================================================
# Local Windows Environment
# ==========================================================

else:

    DATASET_DIR = os.path.join(
        PROJECT_ROOT,
        "dataset"
    )

    RAW_DIR = os.path.join(
        DATASET_DIR,
        "raw"
    )

    SPLIT_DIR = os.path.join(
        DATASET_DIR,
        "split"
    )

    CSV_PATH = os.path.join(
        RAW_DIR,
        "carinthia-s.csv"
    )

# ==========================================================
# Dataset folders
# ==========================================================

TRAIN_IMAGES_DIR = os.path.join(
    SPLIT_DIR,
    "train",
    "images"
)

TRAIN_MASKS_DIR = os.path.join(
    SPLIT_DIR,
    "train",
    "masks"
)

VAL_IMAGES_DIR = os.path.join(
    SPLIT_DIR,
    "val",
    "images"
)

VAL_MASKS_DIR = os.path.join(
    SPLIT_DIR,
    "val",
    "masks"
)

TEST_IMAGES_DIR = os.path.join(
    SPLIT_DIR,
    "test",
    "images"
)

TEST_MASKS_DIR = os.path.join(
    SPLIT_DIR,
    "test",
    "masks"
)
if __name__ == "__main__":

    print("=" * 60)

    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("DATASET_DIR  :", DATASET_DIR)
    print("CSV_PATH     :", CSV_PATH)

    print("CSV EXISTS   :", os.path.exists(CSV_PATH))

    print("TRAIN IMG    :", os.path.exists(TRAIN_IMAGES_DIR))
    print("TRAIN MASK   :", os.path.exists(TRAIN_MASKS_DIR))

    print("VAL IMG      :", os.path.exists(VAL_IMAGES_DIR))
    print("VAL MASK     :", os.path.exists(VAL_MASKS_DIR))

    print("TEST IMG     :", os.path.exists(TEST_IMAGES_DIR))
    print("TEST MASK    :", os.path.exists(TEST_MASKS_DIR))

    print("DEVICE       :", torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    print("=" * 60)