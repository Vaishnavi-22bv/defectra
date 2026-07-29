"""
train.py
DEFECTRA - Multi-Task Learning Training Pipeline
=================================================
Tasks     : Binary Semantic Segmentation + 6-Class Defect Classification
Model     : DefectraUNetPlusPlus (U-Net++ with EfficientNet-B2 encoder)
Optimizer : AdamW
Scheduler : CosineAnnealingLR
Framework : PyTorch 2.x
Python    : 3.11

Run from the project root or backend/ directory:
    python backend/training/train.py
    python training/train.py          # from inside backend/
"""

import os
import sys
import time
import random

# ---------------------------------------------------------------------------
# Resolve backend/ so all sibling packages are importable regardless of the
# working directory from which this script is invoked.
# ---------------------------------------------------------------------------
_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/training/
_BACKEND_DIR  = os.path.dirname(_TRAINING_DIR)               # backend/

for _d in (_BACKEND_DIR, _TRAINING_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config — import ONLY from config.py; never redefine these values.
# ---------------------------------------------------------------------------
from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    BATCH_SIZE,
    NUM_CLASSES,
    CLASS_LABELS,
    NUM_SEG_CLASSES,
    PIN_MEMORY,
    NUM_WORKERS,
    SHUFFLE_TRAIN,
    SEG_LOSS_WEIGHT,
    CLS_LOSS_WEIGHT,
    SEED,
    EARLY_STOPPING_PATIENCE,
)

# ---------------------------------------------------------------------------
# Project modules — names must match the existing files exactly.
# ---------------------------------------------------------------------------
from preprocessing.dataloader import get_dataloaders
from models.unetplusplus       import DefectraUNetPlusPlus
from losses                    import MultiTaskLoss   # same package (training/)


# ===========================================================================
# Reproducibility
# ===========================================================================

def _set_seed(seed: int) -> None:
    """
    Fix Python, NumPy, and PyTorch random seeds for reproducible training.

    Parameters
    ----------
    seed : int
        Seed value read from config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Validation metric helper
# ===========================================================================

def _compute_dice_score(
    seg_logits:  torch.Tensor,
    seg_targets: torch.Tensor,
    threshold:   float = 0.5,
    smooth:      float = 1e-6,
) -> float:
    """
    Compute the mean Dice Score over a batch (metric only — not the loss).

    The model emits raw logits (activation=None), so sigmoid is applied here
    before thresholding.  This mirrors the DiceLoss implementation but is
    used purely for reporting during validation.

    Parameters
    ----------
    seg_logits  : torch.Tensor  shape (B, 1, H, W)  raw model logits
    seg_targets : torch.Tensor  shape (B, 1, H, W)  binary float masks {0, 1}
    threshold   : float         binarisation threshold after sigmoid
    smooth      : float         numerical stability constant

    Returns
    -------
    float : Dice Score in [0, 1]
    """
    with torch.no_grad():
        probs = torch.sigmoid(seg_logits)
        preds = (probs > threshold).float()

        preds   = preds.contiguous().view(-1)
        targets = seg_targets.contiguous().view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    return dice.item()


# ===========================================================================
# train_one_epoch
# ===========================================================================

def train_one_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> dict:
    """
    Execute one complete training epoch.

    For every batch:
        1. Move tensors to device.
        2. Run the forward pass through DefectraUNetPlusPlus.
        3. Compute MultiTaskLoss.
        4. Run backward propagation.
        5. Call optimizer.step().

    Parameters
    ----------
    model     : DefectraUNetPlusPlus in training mode
    loader    : training DataLoader — yields (images, masks, labels)
    criterion : MultiTaskLoss
    optimizer : AdamW
    device    : torch.device — DEVICE from config

    Returns
    -------
    dict
        "total_loss"          : float — mean total loss across all batches
        "segmentation_loss"   : float — mean segmentation loss
        "classification_loss" : float — mean classification loss
    """
    model.train()

    running_total = 0.0
    running_seg   = 0.0
    running_cls   = 0.0
    num_batches   = 0

    for images, masks, labels in loader:
        # ── Move data to device ───────────────────────────────────────────
        images = images.to(device, non_blocking=True)   # (B, 3, 256, 256) float32
        masks  = masks.to(device,  non_blocking=True)   # (B, 1, 256, 256) float32
        labels = labels.to(device, non_blocking=True)   # (B,)             torch.long

        # ── Forward pass ──────────────────────────────────────────────────
        segmentation_output, classification_output = model(images)
        # segmentation_output   : (B, 1, 256, 256)  raw logits
        # classification_output : (B, 6)             raw logits

        # ── Multi-task loss ────────────────────────────────────────────────
        # Positional order must match MultiTaskLoss.forward() exactly:
        #   (seg_pred, seg_target, cls_pred, cls_target)
        loss_dict = criterion(
            segmentation_output,    # predicted segmentation logits
            masks,                  # ground-truth binary masks
            classification_output,  # predicted class logits
            labels,                 # ground-truth class indices
        )

        total_loss = loss_dict["total_loss"]   # not detached — safe for backward()

        # ── Backward propagation + optimiser step ─────────────────────────
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ── Accumulate scalars for epoch-level reporting ───────────────────
        running_total += total_loss.item()
        running_seg   += loss_dict["segmentation_loss"].item()    # already detached
        running_cls   += loss_dict["classification_loss"].item()  # already detached
        num_batches   += 1

    n = max(num_batches, 1)
    return {
        "total_loss":          running_total / n,
        "segmentation_loss":   running_seg   / n,
        "classification_loss": running_cls   / n,
    }


# ===========================================================================
# validate_one_epoch
# ===========================================================================

def validate_one_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    """
    Execute one complete validation epoch (no gradient computation).

    Computes:
        - Multi-task losses (total / segmentation / classification)
        - Dice Score       (segmentation quality metric)
        - Classification Accuracy

    Parameters
    ----------
    model     : DefectraUNetPlusPlus in eval mode
    loader    : validation DataLoader — yields (images, masks, labels)
    criterion : MultiTaskLoss
    device    : torch.device — DEVICE from config

    Returns
    -------
    dict
        "total_loss"          : float — mean total loss
        "segmentation_loss"   : float — mean segmentation loss
        "classification_loss" : float — mean classification loss
        "dice_score"          : float — mean Dice Score  ∈ [0, 1]
        "accuracy"            : float — classification accuracy ∈ [0, 1]
    """
    model.eval()

    running_total = 0.0
    running_seg   = 0.0
    running_cls   = 0.0
    running_dice  = 0.0
    num_batches   = 0

    correct_cls = 0
    total_cls   = 0

    with torch.no_grad():
        for images, masks, labels in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Forward pass ──────────────────────────────────────────────
            segmentation_output, classification_output = model(images)

            # ── Multi-task loss ───────────────────────────────────────────
            loss_dict = criterion(
                segmentation_output,
                masks,
                classification_output,
                labels,
            )

            running_total += loss_dict["total_loss"].item()
            running_seg   += loss_dict["segmentation_loss"].item()
            running_cls   += loss_dict["classification_loss"].item()

            # ── Dice Score (metric, not loss) ─────────────────────────────
            dice = _compute_dice_score(segmentation_output, masks)
            running_dice += dice

            # ── Classification accuracy ───────────────────────────────────
            predicted    = classification_output.argmax(dim=1)       # (B,)
            correct_cls += (predicted == labels).sum().item()
            total_cls   += labels.size(0)

            num_batches += 1

    n = max(num_batches, 1)
    return {
        "total_loss":          running_total / n,
        "segmentation_loss":   running_seg   / n,
        "classification_loss": running_cls   / n,
        "dice_score":          running_dice  / n,
        "accuracy":            correct_cls   / max(total_cls, 1),
    }


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def build_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    best_dice: float,
    best_acc:  float,
) -> dict:
    """
    Assemble a serialisable checkpoint dictionary.

    Captures full training state so that training can be resumed from any
    saved checkpoint without loss of optimiser momentum or scheduler state.

    Parameters
    ----------
    model     : DefectraUNetPlusPlus
    optimizer : AdamW
    scheduler : CosineAnnealingLR
    epoch     : current epoch (1-based)
    best_dice : best validation Dice Score recorded so far
    best_acc  : best validation classification accuracy recorded so far

    Returns
    -------
    dict  — ready for torch.save()
    """
    return {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dice":            best_dice,
        "best_acc":             best_acc,
    }


def save_checkpoint(checkpoint: dict, path: str) -> None:
    """
    Persist a checkpoint dictionary to disk.

    Creates any missing parent directories automatically so this function
    works even if CHECKPOINT_DIR does not yet exist on a fresh clone.

    Parameters
    ----------
    checkpoint : dict  — built by build_checkpoint()
    path       : str   — destination file path (BEST_MODEL_PATH or LAST_MODEL_PATH)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(checkpoint, path)


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    """
    Orchestrate the full DEFECTRA training pipeline.

    Steps
    -----
    1.  Fix random seeds (SEED from config).
    2.  Load train / val / test DataLoaders via get_dataloaders().
    3.  Instantiate DefectraUNetPlusPlus and move to DEVICE.
    4.  Initialise MultiTaskLoss, AdamW, and CosineAnnealingLR.
    5.  For each epoch:
            a. train_one_epoch()
            b. validate_one_epoch()
            c. scheduler.step()
            d. save_checkpoint() → LAST_MODEL_PATH  (every epoch)
            e. save_checkpoint() → BEST_MODEL_PATH  (when Dice Score improves)
            f. print epoch statistics
    6.  Early stopping when EARLY_STOPPING_PATIENCE consecutive epochs pass
        without a new best Dice Score.
    """

    # ── 1. Reproducibility ────────────────────────────────────────────────
    _set_seed(SEED)

    print("=" * 65)
    print("  DEFECTRA MULTI-TASK TRAINING PIPELINE")
    print("=" * 65)
    print(f"  Device                : {DEVICE}")
    print(f"  Epochs                : {NUM_EPOCHS}")
    print(f"  Batch Size            : {BATCH_SIZE}")
    print(f"  Learning Rate         : {LEARNING_RATE}")
    print(f"  Weight Decay          : {WEIGHT_DECAY}")
    print(f"  Seg Loss Weight       : {SEG_LOSS_WEIGHT}")
    print(f"  Cls Loss Weight       : {CLS_LOSS_WEIGHT}")
    print(f"  Early-Stop Patience   : {EARLY_STOPPING_PATIENCE}")
    print(f"  Num Classes (cls)     : {NUM_CLASSES}")
    print(f"  Num Classes (seg)     : {NUM_SEG_CLASSES}")
    print("=" * 65)

    # ── 2. DataLoaders ────────────────────────────────────────────────────
    print("\n[1/4] Loading dataloaders ...")
    train_loader, val_loader, test_loader, label_to_idx = get_dataloaders()
    print(f"      Train samples : {len(train_loader.dataset)}")
    print(f"      Val   samples : {len(val_loader.dataset)}")
    print(f"      Test  samples : {len(test_loader.dataset)}")
    model_label_names = {
        model_label: CLASS_LABELS[dataset_label]
        for dataset_label, model_label in label_to_idx.items()
    }
    print(f"      label_to_idx  : {label_to_idx}")
    print(f"      class_names   : {model_label_names}")

    # ── 3. Model ──────────────────────────────────────────────────────────
    print("\n[2/4] Building model ...")
    model = DefectraUNetPlusPlus().to(DEVICE)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total parameters     : {total_params:,}")
    print(f"      Trainable parameters : {trainable_params:,}")

    # ── 4. Loss / Optimiser / Scheduler ───────────────────────────────────
    print("\n[3/4] Initialising criterion, optimiser, and scheduler ...")
    criterion = MultiTaskLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # T_max = NUM_EPOCHS gives one full cosine cycle over the training run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=0.0,
    )

    # ── 5. Training loop ──────────────────────────────────────────────────
    print("\n[4/4] Starting training ...\n")
    print(
        f"{'Epoch':>7}  {'Time':>6}  {'LR':>9}  "
        f"{'Tr-Total':>9}  {'Tr-Seg':>8}  {'Tr-Cls':>8}  "
        f"{'Va-Total':>9}  {'Va-Seg':>8}  {'Va-Cls':>8}  "
        f"{'Dice':>7}  {'Acc':>7}"
    )
    print("-" * 105)

    best_dice          = 0.0
    best_acc           = 0.0
    epochs_without_imp = 0   # consecutive epochs without Dice improvement

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        # ── Validate ──────────────────────────────────────────────────────
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        # ── Scheduler step (after validation) ─────────────────────────────
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        val_dice   = val_metrics["dice_score"]
        val_acc    = val_metrics["accuracy"]

        # ── Print epoch statistics ─────────────────────────────────────────
        print(
            f"{epoch:>7}/{NUM_EPOCHS:<4}  "
            f"{epoch_time:>5.1f}s  "
            f"{current_lr:>9.2e}  "
            f"{train_metrics['total_loss']:>9.4f}  "
            f"{train_metrics['segmentation_loss']:>8.4f}  "
            f"{train_metrics['classification_loss']:>8.4f}  "
            f"{val_metrics['total_loss']:>9.4f}  "
            f"{val_metrics['segmentation_loss']:>8.4f}  "
            f"{val_metrics['classification_loss']:>8.4f}  "
            f"{val_dice:>7.4f}  "
            f"{val_acc:>7.4f}"
        )

        # ── Save last checkpoint (every epoch) ────────────────────────────
        last_ckpt = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_dice=best_dice,
            best_acc=best_acc,
        )
        save_checkpoint(last_ckpt, LAST_MODEL_PATH)

        # ── Save best checkpoint (primary metric: validation Dice Score) ───
        if val_dice > best_dice:
            best_dice = val_dice
            best_acc  = max(best_acc, val_acc)

            best_ckpt = build_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_dice=best_dice,
                best_acc=best_acc,
            )
            save_checkpoint(best_ckpt, BEST_MODEL_PATH)
            print(
                f"         ✔  New best model saved at epoch {epoch} — "
                f"Dice: {best_dice:.4f}  Acc: {best_acc:.4f}"
            )
            epochs_without_imp = 0
        else:
            epochs_without_imp += 1

        # ── Early stopping check ───────────────────────────────────────────
        if epochs_without_imp >= EARLY_STOPPING_PATIENCE:
            print(
                f"\n  Early stopping triggered at epoch {epoch}. "
                f"No Dice improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print(f"  Best Validation Dice Score   : {best_dice:.4f}")
    print(f"  Best Validation Accuracy     : {best_acc:.4f}")
    print(f"  Best Model Checkpoint        : {BEST_MODEL_PATH}")
    print(f"  Last Model Checkpoint        : {LAST_MODEL_PATH}")
    print("=" * 65)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()
