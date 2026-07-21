"""
=========================================================
DEFECTRA - Multi-Task U-Net++
Semantic Segmentation + Defect Classification

Encoder  : EfficientNet-B2
Framework: segmentation_models_pytorch 0.5.0
=========================================================
"""

import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# The filters are local to the dependency import.  They silence known
# third-party import-time warnings without hiding warnings from this project.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"segmentation_models_pytorch(?:\..*)?",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"timm(?:\..*)?",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.jit\.interface` is deprecated\..*",
        category=DeprecationWarning,
    )
    import segmentation_models_pytorch as smp

# ==========================================================
# Resolve backend directory
# ==========================================================

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ==========================================================
# Read configuration
# ==========================================================

from config import (
    DEVICE,
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
)

# ==========================================================
# Model
# ==========================================================


class DefectraUNetPlusPlus(nn.Module):
    """
    Multi-task U-Net++

    Outputs
    -------
    segmentation_output:
        (B,1,H,W)

    classification_output:
        (B,6)
    """

    def __init__(self):
        super().__init__()

        self.network = smp.UnetPlusPlus(

            encoder_name=ENCODER_NAME,

            encoder_weights=ENCODER_WEIGHTS,

            in_channels=IN_CHANNELS,

            classes=1,

            activation=None,

            aux_params={

                "pooling": "avg",

                "dropout": 0.30,

                "classes": NUM_CLASSES,

                "activation": None,
            },
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        segmentation_output, classification_output = self.network(x)

        return segmentation_output, classification_output


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEFECTRA MULTI-TASK U-NET++")
    print("=" * 60)

    model = DefectraUNetPlusPlus().to(DEVICE)

    model.eval()

    dummy = torch.randn(
        2,
        IN_CHANNELS,
        512,
        512,
        device=DEVICE,
    )

    with torch.no_grad():
        segmentation_output, classification_output = model(dummy)

    print(f"Input Shape                 : {tuple(dummy.shape)}")
    print(f"Segmentation Output Shape   : {tuple(segmentation_output.shape)}")
    print(f"Classification Output Shape : {tuple(classification_output.shape)}")

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    print(f"Device               : {DEVICE}")

    print("=" * 60)
    print("Model Loaded Successfully!")
    print("=" * 60)
