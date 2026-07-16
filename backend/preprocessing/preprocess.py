import os
import cv2
import pandas as pd
from tqdm import tqdm

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_DATASET = os.path.join(BASE_DIR, "dataset", "raw")
CSV_FILE = os.path.join(RAW_DATASET, "carinthia-s.csv")

IMAGE_FOLDER = os.path.join(RAW_DATASET, "images")
MASK_FOLDER = os.path.join(RAW_DATASET, "masks")

PROCESSED_IMAGES = os.path.join(BASE_DIR, "dataset", "processed", "images")
PROCESSED_MASKS = os.path.join(BASE_DIR, "dataset", "processed", "masks")

IMAGE_SIZE = (256, 256)

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

os.makedirs(PROCESSED_IMAGES, exist_ok=True)
os.makedirs(PROCESSED_MASKS, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("DEFECTRA IMAGE PREPROCESSING")
print("=" * 60)

df = pd.read_csv(CSV_FILE, sep=";")

print(f"\nTotal Images : {len(df)}")

processed = 0
skipped = 0

# ==========================================================
# PROCESS DATASET
# ==========================================================

for _, row in tqdm(df.iterrows(), total=len(df)):

    image_path = os.path.join(RAW_DATASET, row["image_path"])
    mask_path = os.path.join(RAW_DATASET, row["mask_path"])

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        skipped += 1
        continue

    # Resize
    image = cv2.resize(image, IMAGE_SIZE)
    mask = cv2.resize(mask, IMAGE_SIZE)

    # Normalize image
    image = image.astype("float32") / 255.0

    # Get original file name from image path
    filename = os.path.basename(image_path)

    # Save processed image
    image_save = os.path.join(PROCESSED_IMAGES, filename)
    mask_save = os.path.join(PROCESSED_MASKS, filename)

    image_uint8 = (image * 255).astype("uint8")

    cv2.imwrite(image_save, image_uint8)
    cv2.imwrite(mask_save, mask)

    processed += 1

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"Processed Images : {processed}")
print(f"Skipped Images   : {skipped}")

print(f"\nProcessed Images Folder : {PROCESSED_IMAGES}")
print(f"Processed Masks Folder  : {PROCESSED_MASKS}")