import os
import pandas as pd

# ==========================
# Dataset Paths
# ==========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")

CSV_FILE = os.path.join(DATASET_DIR, "carinthia-s.csv")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")


def main():

    print("=" * 50)
    print("DEFECTRA DATASET INFORMATION")
    print("=" * 50)

    # Check CSV
    if not os.path.exists(CSV_FILE):
        print("❌ CSV file not found!")
        return

    print(f"\nCSV File : {CSV_FILE}")

    # Read CSV (Carinthia-S uses semicolon as separator)
    df = pd.read_csv(CSV_FILE, sep=";")

    print("\n✅ Dataset Loaded Successfully!")

    print(f"\nNumber of Samples : {len(df)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst Five Rows:")
    print(df.head())

    print("\nImages Folder Exists :", os.path.exists(IMAGE_DIR))
    print("Masks Folder Exists  :", os.path.exists(MASK_DIR))

    print(f"\nNumber of Images : {len(os.listdir(IMAGE_DIR))}")
    print(f"Number of Masks  : {len(os.listdir(MASK_DIR))}")

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDefect Class Distribution:")

    if "label" in df.columns:
        print(df["label"].value_counts())
    else:
        print("Label column not found.")

    print("\nDataset Summary:")
    print(df.describe(include="all"))

    print("\n✅ Dataset Analysis Completed Successfully!")


if __name__ == "__main__":
    main()