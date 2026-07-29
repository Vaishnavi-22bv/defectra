"""
Run Defectra inference on the sample image.

Usage:
    python predictions/predict.py
"""

from pathlib import Path
import argparse
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models.unetplusplus import DefectraUNetPlusPlus
from config import (
    ACTIVE_DATASET_LABELS,
    CLASS_LABELS,
    IMAGE_SIZE,
    MEAN,
    OUTPUT_DIR as CONFIG_OUTPUT_DIR,
    STD,
)

MODEL_PATH = BACKEND_DIR / "saved_models" / "best_model.pth"

DEFAULT_IMAGE_PATH = PROJECT_DIR / "test_images" / "9bd62f06fc9e4f2cbe707c229265297d.jpg"
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

# Keep generated inference files separate from training checkpoints and logs.
OUTPUT_DIR = Path(CONFIG_OUTPUT_DIR) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Defect Classes
# --------------------------------------------------

CLASS_NAMES = [CLASS_LABELS[label] for label in ACTIVE_DATASET_LABELS]

# --------------------------------------------------
# Load Model
# --------------------------------------------------

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    if all(k.startswith("model.") for k in state_dict):

        state_dict = {
            f"network.{k.removeprefix('model.')}" : v
            for k,v in state_dict.items()
        }

    model = DefectraUNetPlusPlus()

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        raise RuntimeError(
            "This checkpoint was trained for a different set of classes. "
            "Retrain the model after changing the active labels before running prediction."
        ) from None

    model.to(DEVICE)

    model.eval()

    return model


def get_image_path() -> Path:
    """Read an optional image argument, falling back to a test image."""
    parser = argparse.ArgumentParser(description="Run Defectra inference on an image.")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Path to the image to analyze (defaults to the first file in test_images).",
    )
    args = parser.parse_args()

    if args.image is not None:
        return args.image.expanduser().resolve()

    if DEFAULT_IMAGE_PATH.is_file():
        return DEFAULT_IMAGE_PATH

    test_images_dir = PROJECT_DIR / "test_images"
    candidates = sorted(
        path for path in test_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ) if test_images_dir.is_dir() else []

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No supported image found in {test_images_dir}. "
        "Pass an image path, for example: python predictions/predict.py path/to/image.png"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading Model...")

    model = load_model()

    print("Model Loaded Successfully")

    image_path = get_image_path()
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read input image: {image_path}")

    original = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Keep inference identical to validation/training preprocessing.  The
    # EfficientNet encoder was trained on ImageNet-normalised 512px inputs;
    # sending unnormalised 256px images produces unreliable predictions.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        segmentation_output, classification_output = model(input_tensor)

    print("Prediction Completed")

    # ---------------------------------------------
    # Segmentation Mask
    # ---------------------------------------------

    mask = torch.sigmoid(segmentation_output)

    mask = mask.squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255

    mask_path = OUTPUT_DIR / "predicted_mask.png"

    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Unable to save predicted mask: {mask_path}")

    # ---------------------------------------------
    # Overlay
    # ---------------------------------------------

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))

    overlay = cv2.addWeighted(
        original,
        0.7,
        mask_color,
        0.3,
        0
    )

    overlay_path = OUTPUT_DIR / "overlay.png"

    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Unable to save overlay image: {overlay_path}")

    # ---------------------------------------------
    # Classification
    # ---------------------------------------------

    probs = torch.softmax(classification_output,dim=1)

    confidence,pred = torch.max(probs,1)

    defect = CLASS_NAMES[pred.item()]

    # The model uses compact zero-based indices; show the original CSV label.
    dataset_label = ACTIVE_DATASET_LABELS[pred.item()]

    confidence = confidence.item()*100

    # ---------------------------------------------
    # Area
    # ---------------------------------------------

    area = (mask>0).sum()

    total = mask.size

    percentage = (area/total)*100

    # ---------------------------------------------
    # Results
    # ---------------------------------------------

    print("\n========== RESULT ==========")

    print(f"Defect           : {defect}")

    print(f"Dataset Label    : {dataset_label}")

    print(f"Confidence       : {confidence:.2f}%")

    print(f"Affected Area    : {percentage:.2f}%")

    print(f"\nMask Saved       : {mask_path}")

    print(f"Overlay Saved    : {overlay_path}")

    print("============================")


if __name__=="__main__":
    main()
