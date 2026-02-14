"""
train_mlp.py — Train the Landmark-Based MLP Classifier
=========================================================
Trains gesture/mlp_model.py on collected landmark data.

Expected data format (CSV or .npy):
    Each row = 42 floats (features) + 1 int (label index)
    Saved in  dataset/landmarks_train.npy  and  dataset/landmarks_val.npy

Usage:
    python -m gesture.train_mlp
    python -m gesture.train_mlp --epochs 30 --lr 0.001
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NUM_CLASSES, FEATURE_DIM, MODEL_PATH, GESTURE_LABELS
from gesture.mlp_model import GestureMLP


DATASET_DIR = "dataset"
TRAIN_FILE = os.path.join(DATASET_DIR, "landmarks_train.npy")
VAL_FILE = os.path.join(DATASET_DIR, "landmarks_val.npy")


def load_data(path):
    """Load an .npy file → (features, labels) tensors."""
    data = np.load(path).astype(np.float32)
    X = torch.from_numpy(data[:, :FEATURE_DIM])
    y = torch.from_numpy(data[:, FEATURE_DIM]).long()
    return X, y


def train(epochs=25, batch_size=64, lr=0.001):
    if not os.path.exists(TRAIN_FILE):
        print(f"[train_mlp] Training data not found at {TRAIN_FILE}")
        print("  Collect landmark data first.")
        return

    X_train, y_train = load_data(TRAIN_FILE)
    print(f"[train_mlp] Train samples: {len(X_train)}")

    has_val = os.path.exists(VAL_FILE)
    if has_val:
        X_val, y_val = load_data(VAL_FILE)
        print(f"[train_mlp] Val   samples: {len(X_val)}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True
    )

    model = GestureMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*50}")
    print(f"  Training GestureMLP — {epochs} epochs")
    print(f"  Classes: {GESTURE_LABELS}")
    print(f"{'='*50}\n")

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            out = model(X_batch)
            loss = criterion(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct += (out.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        train_acc = 100.0 * correct / total

        # Validation
        val_str = ""
        if has_val:
            model.eval()
            with torch.no_grad():
                out = model(X_val)
                val_acc = 100.0 * (out.argmax(1) == y_val).sum().item() / len(y_val)
            val_str = f"  Val Acc: {val_acc:6.2f}%"

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
                val_str += "  ← saved"
        else:
            # Save every epoch if no val set
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"  Epoch [{epoch:2d}/{epochs}]  "
              f"Loss: {total_loss/total:.4f}  "
              f"Train Acc: {train_acc:6.2f}%{val_str}")

    print(f"\n  Done. Model saved to {MODEL_PATH}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ATLAS GestureMLP")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
