"""
==========================================
DEFECTRA LOSS FUNCTIONS
Multi-Task Learning
(Binary Segmentation + 6-Class Classification)
==========================================
"""

import os
import sys

# ---------------------------------------------------------
# Fix config import
# ---------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

import torch
import torch.nn as nn

from config import (
    SEG_LOSS_WEIGHT,
    CLS_LOSS_WEIGHT,
)

# ---------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss for Binary Segmentation
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        predictions = torch.sigmoid(predictions)
        predictions = predictions.clamp(1e-7, 1.0 - 1e-7)

        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        smooth = 1e-6

        intersection = (predictions * targets).sum()

        dice = (
            (2.0 * intersection + smooth)
            /
            (predictions.sum() + targets.sum() + smooth)
        )

        return 1.0 - dice


# ---------------------------------------------------------
# Segmentation Loss
# BCE + Dice
# ---------------------------------------------------------

class SegmentationLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        bce_loss = self.bce(predictions, targets)

        dice_loss = self.dice(predictions, targets)

        return bce_loss + dice_loss


# ---------------------------------------------------------
# Classification Loss
# ---------------------------------------------------------

class ClassificationLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        return self.ce(predictions, labels)


# ---------------------------------------------------------
# Multi-Task Loss
# ---------------------------------------------------------

class MultiTaskLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.segmentation_loss = SegmentationLoss()
        self.classification_loss = ClassificationLoss()

    def forward(

        self,

        segmentation_output: torch.Tensor,
        segmentation_target: torch.Tensor,

        classification_output: torch.Tensor,
        classification_target: torch.Tensor,

    ):

        seg_loss = self.segmentation_loss(

            segmentation_output,
            segmentation_target,

        )

        cls_loss = self.classification_loss(

            classification_output,
            classification_target,

        )

        total_loss = (

            SEG_LOSS_WEIGHT * seg_loss +

            CLS_LOSS_WEIGHT * cls_loss

        )

        return {

            "total_loss": total_loss,

            "segmentation_loss": seg_loss.detach(),

            "classification_loss": cls_loss.detach(),

        }


# ---------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("DEFECTRA MULTI-TASK LOSS FUNCTION")
    print("=" * 55)

    batch_size = 4

    # Binary Segmentation
    seg_predictions = torch.randn(batch_size, 1, 256, 256)

    seg_targets = torch.randint(
        0,
        2,
        (batch_size, 1, 256, 256)
    ).float()

    # Classification (6 classes)
    cls_predictions = torch.randn(batch_size, 6)

    cls_targets = torch.randint(
        0,
        6,
        (batch_size,)
    )

    criterion = MultiTaskLoss()

    losses = criterion(

        seg_predictions,
        seg_targets,

        cls_predictions,
        cls_targets,

    )

    print(f"Total Loss          : {losses['total_loss'].item():.4f}")
    print(f"Segmentation Loss   : {losses['segmentation_loss'].item():.4f}")
    print(f"Classification Loss : {losses['classification_loss'].item():.4f}")

    print("=" * 55)