"""
DEFECTRA Configuration File
Compatible with:
- VS Code (Windows)
- Kaggle Notebook
"""

import os
import torch

# ==========================================================
# Project Paths
# ==========================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# ==========================================================
# Dataset Paths
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
# Image Parameters
# ==========================================================

IMAGE_SIZE = 512

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ==========================================================
# Dataset Parameters
# ==========================================================

NUM_CLASSES = 6
NUM_SEG_CLASSES = 1

# ==========================================================
# Model Architecture
# ==========================================================

ENCODER_NAME = "efficientnet-b2"
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3

# ==========================================================
# Training Parameters
# ==========================================================

BATCH_SIZE = 8
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 2

SHUFFLE_TRAIN = True

SEG_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 1.0

EARLY_STOPPING_PATIENCE = 10

SEED = 42

# ==========================================================
# Data Augmentation
# ==========================================================

USE_AUGMENTATION = True

TRAIN_AUG = {

    "horizontal_flip_prob": 0.5,

    "vertical_flip_prob": 0.5,

    "random_rotate90_prob": 0.5,

    "blur_limit": 3,

    "rotate_limit": 15

}

VAL_AUG = {}

TEST_AUG = {}

# ==========================================================
# Model Save Paths
# ==========================================================

BEST_MODEL_PATH = os.path.join(
    SAVE_MODEL_DIR,
    "best_model.pth"
)

LAST_MODEL_PATH = os.path.join(
    SAVE_MODEL_DIR,
    "last_model.pth"
)

CHECKPOINT_PATH = os.path.join(
    SAVE_MODEL_DIR,
    "checkpoint.pth"
)

TRAIN_LOG_PATH = os.path.join(
    LOG_DIR,
    "train_log.csv"
)

VAL_LOG_PATH = os.path.join(
    LOG_DIR,
    "val_log.csv"
)

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
    print("IMAGE_SIZE   :", IMAGE_SIZE)
    print("BATCH_SIZE   :", BATCH_SIZE)
    print("NUM_EPOCHS   :", NUM_EPOCHS)
    print("NUM_CLASSES  :", NUM_CLASSES)

    print("=" * 60)
