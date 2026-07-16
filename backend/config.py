"""
config.py
DEFECTRA - Multi-Task Learning (Semantic Segmentation + Defect Classification)
Carinthia Semiconductor Dataset
"""

import os
import torch

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# defectra project root
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
SPLIT_DIR = os.path.join(DATASET_DIR, "split")

CSV_PATH = os.path.join(RAW_DIR, "carinthia-s.csv")

TRAIN_IMAGES_DIR = os.path.join(SPLIT_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(SPLIT_DIR, "train", "masks")

VAL_IMAGES_DIR = os.path.join(SPLIT_DIR, "val", "images")
VAL_MASKS_DIR = os.path.join(SPLIT_DIR, "val", "masks")

TEST_IMAGES_DIR = os.path.join(SPLIT_DIR, "test", "images")
TEST_MASKS_DIR = os.path.join(SPLIT_DIR, "test", "masks")

# ---------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------

CSV_IMAGE_COL = "image_path"
CSV_MASK_COL = "mask_path"
CSV_FILENAME_COL = "filename"
CSV_LABEL_COL = "label"

# ---------------------------------------------------------------------
# Output / checkpoint paths
# ---------------------------------------------------------------------

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "defectra_best.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "defectra_last.pth")

# ---------------------------------------------------------------------
# Dataset / Task Settings
# ---------------------------------------------------------------------

NUM_CLASSES = 6                 # Classification classes
NUM_SEG_CLASSES = 2             # Binary segmentation

IMAGE_SIZE = 256
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IN_CHANNELS = 3

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

ENCODER_NAME = "efficientnet-b2"
ENCODER_WEIGHTS = "imagenet"
DECODER_NAME = "unetplusplus"
ACTIVATION = None

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

BATCH_SIZE = 8
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

OPTIMIZER = "adamw"
SCHEDULER = "cosine"

NUM_WORKERS = 4

SHUFFLE_TRAIN = True

SEG_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 1.0

EARLY_STOPPING_PATIENCE = 10
SAVE_BEST_ONLY = True

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable pin memory only when using CUDA
PIN_MEMORY = DEVICE.type == "cuda"

# ---------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------

SEED = 42
# ---------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------

MASK_INTERPOLATION = 0
IMAGE_INTERPOLATION = 1

MASK_FILL_VALUE = 0
IMAGE_FILL_VALUE = 0

TRAIN_AUG = {
    "resize": {
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
    },

    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "random_rotate90_prob": 0.5,

    "shift_scale_rotate": {
        "prob": 0.5,
        "shift_limit": 0.0625,
        "scale_limit": 0.10,
        "rotate_limit": 20,
        "border_mode": 0,
    },

    "random_crop": {
        "enabled": False,
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
        "prob": 0.5,
    },

    "brightness_contrast": {
        "prob": 0.5,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
    },

    "hue_saturation_value": {
        "prob": 0.3,
        "hue_shift_limit": 10,
        "sat_shift_limit": 15,
        "val_shift_limit": 10,
    },

    "gauss_noise": {
        "prob": 0.2,
        "var_limit": (10.0, 50.0),
    },

    "gaussian_blur": {
        "prob": 0.2,
        "blur_limit": (3, 5),
    },

    "coarse_dropout": {
        "prob": 0.2,
        "max_holes": 8,
        "max_height": 16,
        "max_width": 16,
        "min_holes": 1,
        "min_height": 4,
        "min_width": 4,
        "fill_value": 0,
    },

    "normalize": {
        "mean": MEAN,
        "std": STD,
        "max_pixel_value": 255.0,
    },
}

VAL_AUG = {
    "resize": {
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
    },

    "normalize": {
        "mean": MEAN,
        "std": STD,
        "max_pixel_value": 255.0,
    },
}

TEST_AUG = VAL_AUG

USE_AUGMENTATION = True

# ---------------------------------------------------------------------
# Debug (Runs only when config.py is executed directly)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("BACKEND_DIR  :", BACKEND_DIR)
    print("DATASET_DIR  :", DATASET_DIR)
    print("RAW_DIR      :", RAW_DIR)
    print("SPLIT_DIR    :", SPLIT_DIR)
    print("CSV_PATH     :", CSV_PATH)
    print("CSV EXISTS   :", os.path.exists(CSV_PATH))
    print("DEVICE       :", DEVICE)
    print("PIN_MEMORY   :", PIN_MEMORY)
    print("=" * 60)