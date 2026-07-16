""""
=========================================================
DEFECTRA METRICS
Multi-Task Evaluation
=========================================================

Segmentation Metrics
--------------------
• Dice Score
• IoU Score
• Pixel Accuracy
• Precision
• Recall
• F1 Score

Classification Metrics
----------------------
• Accuracy
• Precision
• Recall
• F1 Score
=========================================================
"""

import torch


# ==========================================================
# Helper
# ==========================================================

def _binarize(predictions, threshold=0.5):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    return predictions


# ==========================================================
# SEGMENTATION METRICS
# ==========================================================

def dice_score(predictions, targets, threshold=0.5):

    predictions = _binarize(predictions, threshold)

    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    smooth = 1e-6

    intersection = (predictions * targets).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        predictions.sum() + targets.sum() + smooth
    )

    return dice.item()


def iou_score(predictions, targets, threshold=0.5):

    predictions = _binarize(predictions, threshold)

    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    smooth = 1e-6

    intersection = (predictions * targets).sum()

    union = predictions.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def pixel_accuracy(predictions, targets, threshold=0.5):

    predictions = _binarize(predictions, threshold)

    correct = (predictions == targets).float().sum()

    accuracy = correct / targets.numel()

    return accuracy.item()


def precision_score(predictions, targets, threshold=0.5):

    predictions = _binarize(predictions, threshold)

    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    tp = ((predictions == 1) & (targets == 1)).sum().float()
    fp = ((predictions == 1) & (targets == 0)).sum().float()

    precision = tp / (tp + fp + 1e-6)

    return precision.item()


def recall_score(predictions, targets, threshold=0.5):

    predictions = _binarize(predictions, threshold)

    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    tp = ((predictions == 1) & (targets == 1)).sum().float()
    fn = ((predictions == 0) & (targets == 1)).sum().float()

    recall = tp / (tp + fn + 1e-6)

    return recall.item()


def f1_score(predictions, targets, threshold=0.5):

    precision = precision_score(predictions, targets, threshold)
    recall = recall_score(predictions, targets, threshold)

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall + 1e-6
    )

    return f1


# ==========================================================
# CLASSIFICATION METRICS
# ==========================================================

def classification_accuracy(predictions, labels):

    predicted = torch.argmax(predictions, dim=1)

    accuracy = (predicted == labels).float().mean()

    return accuracy.item()


def classification_precision(predictions, labels):

    predicted = torch.argmax(predictions, dim=1)

    num_classes = predictions.size(1)

    precision = []

    for cls in range(num_classes):

        tp = ((predicted == cls) & (labels == cls)).sum().float()

        fp = ((predicted == cls) & (labels != cls)).sum().float()

        precision.append(tp / (tp + fp + 1e-6))

    return torch.mean(torch.stack(precision)).item()


def classification_recall(predictions, labels):

    predicted = torch.argmax(predictions, dim=1)

    num_classes = predictions.size(1)

    recall = []

    for cls in range(num_classes):

        tp = ((predicted == cls) & (labels == cls)).sum().float()

        fn = ((predicted != cls) & (labels == cls)).sum().float()

        recall.append(tp / (tp + fn + 1e-6))

    return torch.mean(torch.stack(recall)).item()


def classification_f1(predictions, labels):

    precision = classification_precision(predictions, labels)

    recall = classification_recall(predictions, labels)

    return (
        2 * precision * recall
    ) / (
        precision + recall + 1e-6
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DEFECTRA METRICS")
    print("=" * 60)

    # -----------------------------
    # Segmentation
    # -----------------------------

    seg_predictions = torch.randn(2, 1, 256, 256)

    seg_targets = torch.randint(
        0,
        2,
        (2, 1, 256, 256)
    ).float()

    print("\nSEGMENTATION")

    print(f"Dice Score     : {dice_score(seg_predictions, seg_targets):.4f}")
    print(f"IoU Score      : {iou_score(seg_predictions, seg_targets):.4f}")
    print(f"Pixel Accuracy : {pixel_accuracy(seg_predictions, seg_targets):.4f}")
    print(f"Precision      : {precision_score(seg_predictions, seg_targets):.4f}")
    print(f"Recall         : {recall_score(seg_predictions, seg_targets):.4f}")
    print(f"F1 Score       : {f1_score(seg_predictions, seg_targets):.4f}")

    # -----------------------------
    # Classification
    # -----------------------------

    cls_predictions = torch.randn(4, 6)

    cls_labels = torch.randint(
        0,
        6,
        (4,)
    )

    print("\nCLASSIFICATION")

    print(f"Accuracy       : {classification_accuracy(cls_predictions, cls_labels):.4f}")
    print(f"Precision      : {classification_precision(cls_predictions, cls_labels):.4f}")
    print(f"Recall         : {classification_recall(cls_predictions, cls_labels):.4f}")
    print(f"F1 Score       : {classification_f1(cls_predictions, cls_labels):.4f}")

    print("=" * 60)