"""
=========================================================
DEFECTRA DATASET SPLITTING
Train : Validation : Test = 60 : 20 : 20
=========================================================
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "..", "dataset", "processed", "images")
MASK_DIR = os.path.join(BASE_DIR, "..", "dataset", "processed", "masks")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "dataset", "split")

# =========================================================
# CREATE FOLDERS
# =========================================================

folders = [

    "train/images",
    "train/masks",

    "val/images",
    "val/masks",

    "test/images",
    "test/masks"

]

for folder in folders:

    os.makedirs(
        os.path.join(OUTPUT_DIR, folder),
        exist_ok=True
    )

# =========================================================
# READ IMAGE LIST
# =========================================================

images = sorted(os.listdir(IMAGE_DIR))

print("=" * 60)
print("DEFECTRA DATASET SPLITTING")
print("=" * 60)

print(f"Total Images : {len(images)}")

# =========================================================
# SPLIT 60 : 20 : 20
# =========================================================

train_images, temp_images = train_test_split(

    images,
    test_size=0.40,
    random_state=42,
    shuffle=True

)

val_images, test_images = train_test_split(

    temp_images,
    test_size=0.50,
    random_state=42,
    shuffle=True

)

print(f"Training Images   : {len(train_images)}")
print(f"Validation Images : {len(val_images)}")
print(f"Testing Images    : {len(test_images)}")


# =========================================================
# COPY FUNCTION
# =========================================================

def copy_files(file_list, split):

    image_output = os.path.join(
        OUTPUT_DIR,
        split,
        "images"
    )

    mask_output = os.path.join(
        OUTPUT_DIR,
        split,
        "masks"
    )

    for image_name in tqdm(file_list):

        image_src = os.path.join(
            IMAGE_DIR,
            image_name
        )

        mask_src = os.path.join(
            MASK_DIR,
            image_name
        )

        shutil.copy2(
            image_src,
            os.path.join(image_output, image_name)
        )

        shutil.copy2(
            mask_src,
            os.path.join(mask_output, image_name)
        )


# =========================================================
# COPY DATA
# =========================================================

print("\nCopying Training Images...")
copy_files(train_images, "train")

print("\nCopying Validation Images...")
copy_files(val_images, "val")

print("\nCopying Testing Images...")
copy_files(test_images, "test")

# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 60)
print("DATASET SPLITTING COMPLETED")
print("=" * 60)

print(f"Train Images      : {len(train_images)}")
print(f"Validation Images : {len(val_images)}")
print(f"Test Images       : {len(test_images)}")

print("\nDataset Saved At")

print(OUTPUT_DIR)