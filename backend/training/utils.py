"""
==========================================
DEFECTRA TRAINING UTILITIES
==========================================
"""

import os
import torch


def create_directory(path):
    """
    Create directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save model checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, save_path)

    print(f"Model saved to: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load saved checkpoint.
    """

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss


if __name__ == "__main__":

    print("=" * 50)
    print("DEFECTRA UTILITIES")
    print("=" * 50)

    create_directory("../saved_models")

    print("Utilities are working successfully!")