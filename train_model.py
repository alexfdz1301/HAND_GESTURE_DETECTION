"""
train_model.py — CNN Training Pipeline
========================================
Trains the GestureCNN on the custom dataset collected by collect_data.py.
Includes moderate data augmentation, label smoothing, and model saving.

Usage:
    python -m gesture.train_model
    python -m gesture.train_model --epochs 15 --batch_size 32

Prerequisites:
    1. Run collect_data.py to build dataset/train/ and dataset/val/
    2. Each class folder should have ~800 (train) and ~200 (val) images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gesture.gesture_model import GestureCNN
from gesture.gesture_labels import NUM_CLASSES, GESTURE_LABELS


# ---------- configuration ----------

DATASET_ROOT = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_cnn.pth")
IMAGE_SIZE = 128


def get_transforms():
    """
    Create training and validation transforms.
    Training includes moderate data augmentation for robustness
    without over-distorting grayscale hand crops.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # ----- moderate augmentation -----
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.3),
        # ----- convert and normalize -----
        transforms.ToTensor(),          # [0, 255] → [0.0, 1.0]
        transforms.Normalize([0.5], [0.5]),  # center around 0
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    return train_transform, val_transform


def train(epochs=15, batch_size=32, learning_rate=0.0005):
    """
    Full training loop for the GestureCNN.

    Args:
        epochs:        number of training epochs (default: 15)
        batch_size:    mini-batch size (default: 32)
        learning_rate: Adam learning rate (default: 0.0005)
    """
    # ----- device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ----- transforms -----
    train_tf, val_tf = get_transforms()

    # ----- datasets -----
    train_dir = os.path.join(DATASET_ROOT, "train")
    val_dir = os.path.join(DATASET_ROOT, "val")

    if not os.path.isdir(train_dir):
        print(f"[ERROR] Training data not found at '{train_dir}'.")
        print("  Run collect_data.py first to create the dataset.")
        return

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_tf)

    print(f"[train] Training samples : {len(train_dataset)}")
    print(f"[train] Validation samples: {len(val_dataset)}")
    print(f"[train] Classes detected  : {train_dataset.classes}")

    # Check that classes match expected labels
    if len(train_dataset.classes) != NUM_CLASSES:
        print(f"[WARNING] Expected {NUM_CLASSES} classes, "
              f"found {len(train_dataset.classes)}.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # ----- model, loss, optimizer -----
    model = GestureCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n{'='*50}")
    print(f"  Training GestureCNN — {epochs} epochs")
    print(f"  Batch size: {batch_size} | LR: {learning_rate}")
    print(f"  Label smoothing: 0.1")
    print(f"{'='*50}\n")

    best_val_acc = 0.0

    # ----- training loop -----
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        print(f"  Epoch [{epoch:2d}/{epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:6.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:6.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"    → Saved best model (val acc: {val_acc:.2f}%)")

    print(f"\n{'='*50}")
    print(f"  Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"{'='*50}\n")


# ---------- CLI entry point ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the ATLAS GestureCNN on collected hand images."
    )
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate for Adam (default: 0.0005)")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size,
          learning_rate=args.lr)
