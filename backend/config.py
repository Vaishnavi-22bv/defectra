"""
==========================================================
DEFECTRA CONFIGURATION
==========================================================
Project  : Defectra
Framework: PyTorch
Model    : U-Net++ + EfficientNet-B2
Dataset  : Carinthia Semiconductor Dataset
==========================================================
"""

import os
import torch

# ==========================================================
# Project Paths
# ==========================================================

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

    CSV_PATH = os.path.join(
        RAW_DIR,
        "carinthia-s.csv"
    )

    SPLIT_DIR = os.path.join(
        DATASET_DIR,
        "split"
    )

# ==========================================================
# Dataset Directories
# ==========================================================

TRAIN_IMAGES_DIR = os.path.join(SPLIT_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(SPLIT_DIR, "train", "masks")

VAL_IMAGES_DIR = os.path.join(SPLIT_DIR, "val", "images")
VAL_MASKS_DIR = os.path.join(SPLIT_DIR, "val", "masks")

TEST_IMAGES_DIR = os.path.join(SPLIT_DIR, "test", "images")
TEST_MASKS_DIR = os.path.join(SPLIT_DIR, "test", "masks")

# ==========================================================
# Output Directories
# ==========================================================

SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PIN_MEMORY = DEVICE.type == "cuda"

# ==========================================================
# Image Settings
# ==========================================================

IMAGE_SIZE = 512

MEAN = (
    0.485,
    0.456,
    0.406,
)

STD = (
    0.229,
    0.224,
    0.225,
)

# ==========================================================
# Dataset Settings
# ==========================================================

NUM_CLASSES = 6

NUM_SEG_CLASSES = 2

# ==========================================================
# DataLoader
# ==========================================================

BATCH_SIZE = 8

NUM_WORKERS = 2

SHUFFLE_TRAIN = True

USE_AUGMENTATION = True

# ==========================================================
# Training
# ==========================================================

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 10

SEG_LOSS_WEIGHT = 1.0

CLS_LOSS_WEIGHT = 1.0

# ==========================================================
# Training Augmentations
# ==========================================================

TRAIN_AUG = {

    "horizontal_flip_prob": 0.5,

    "vertical_flip_prob": 0.5,

    "random_rotate90_prob": 0.5,

    "rotate_limit": 15,

    "blur_limit": 5,

}

VAL_AUG = {}

TEST_AUG = {}

# ==========================================================
# Debug
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DEFECTRA CONFIGURATION")
    print("=" * 60)

    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("DATASET_DIR  :", DATASET_DIR)
    print("CSV_PATH     :", CSV_PATH)

    print()

    print("CSV EXISTS   :", os.path.exists(CSV_PATH))

    print()

    print("TRAIN IMG    :", os.path.exists(TRAIN_IMAGES_DIR))
    print("TRAIN MASK   :", os.path.exists(TRAIN_MASKS_DIR))

    print("VAL IMG      :", os.path.exists(VAL_IMAGES_DIR))
    print("VAL MASK     :", os.path.exists(VAL_MASKS_DIR))

    print("TEST IMG     :", os.path.exists(TEST_IMAGES_DIR))
    print("TEST MASK    :", os.path.exists(TEST_MASKS_DIR))

    print()

    print("DEVICE       :", DEVICE)
    print("PIN_MEMORY   :", PIN_MEMORY)

    print()

    print("IMAGE_SIZE   :", IMAGE_SIZE)
    print("BATCH_SIZE   :", BATCH_SIZE)
    print("NUM_CLASSES  :", NUM_CLASSES)
    print("EPOCHS       :", NUM_EPOCHS)

    print("=" * 60)