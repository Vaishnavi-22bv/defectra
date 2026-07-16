"""
=========================================================
DEFECTRA MULTI-TASK U-NET++
Semantic Segmentation + Defect Classification
Compatible with segmentation_models_pytorch 0.5.0
=========================================================
"""

import os
import sys
import warnings

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# ---------------------------------------------------------
# Add backend directory
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import (
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    DEVICE,
)

# ---------------------------------------------------------
# Ignore harmless warnings
# ---------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"segmentation_models_pytorch.*"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"timm.*"
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)


# =========================================================
# DEFECTRA MODEL
# =========================================================

class DefectraUNetPlusPlus(nn.Module):

    """
    Multi-task U-Net++

    Task 1:
        Binary Semantic Segmentation

    Task 2:
        Six-Class Classification
    """

    def __init__(self):

        super().__init__()

        self.model = smp.UnetPlusPlus(

            encoder_name=ENCODER_NAME,

            encoder_weights=ENCODER_WEIGHTS,

            in_channels=IN_CHANNELS,

            classes=1,

            activation=None,

            aux_params=dict(

                pooling="avg",

                dropout=0.30,

                classes=NUM_CLASSES,

                activation=None,

            ),

        )

    def forward(self, x):

        segmentation_output, classification_output = self.model(x)

        return segmentation_output, classification_output


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DEFECTRA MULTI-TASK U-NET++")
    print("=" * 60)

    model = DefectraUNetPlusPlus().to(DEVICE)

    model.eval()

    dummy = torch.randn(
        2,
        IN_CHANNELS,
        256,
        256,
        device=DEVICE
    )

    with torch.no_grad():

        seg_output, cls_output = model(dummy)

    print()

    print("Input Shape                 :", dummy.shape)

    print("Segmentation Output Shape   :", seg_output.shape)

    print("Classification Output Shape :", cls_output.shape)

    print()

    total_params = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print("Total Parameters     :", f"{total_params:,}")

    print("Trainable Parameters :", f"{trainable_params:,}")

    print()

    print("Device :", DEVICE)

    print("Model Created Successfully!")

    print("=" * 60)