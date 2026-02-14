"""
mlp_model.py — MLP Gesture Classifier
========================================
Lightweight feed-forward network for landmark-based gesture classification.

Architecture:  42 → 64 (ReLU, Dropout) → 32 (ReLU, Dropout) → 7
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, FEATURE_DIM, MLP_HIDDEN_1, MLP_HIDDEN_2, MLP_DROPOUT, MODEL_PATH
from gesture.gesture_labels import get_label


class GestureMLP(nn.Module):
    """Simple 3-layer MLP for gesture classification from landmark features."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, MLP_HIDDEN_1)
        self.fc2 = nn.Linear(MLP_HIDDEN_1, MLP_HIDDEN_2)
        self.fc3 = nn.Linear(MLP_HIDDEN_2, NUM_CLASSES)
        self.drop = nn.Dropout(MLP_DROPOUT)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)   # raw logits


class GestureClassifier:
    """
    Wrapper: loads trained MLP weights, runs single-sample inference.
    Returns (label_str, confidence_float).
    """

    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        self.model = GestureMLP().to(self.device)
        self.model_path = model_path or MODEL_PATH
        self.loaded = False

        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            self.loaded = True
            print(f"[MLP] ✅ Model loaded from {self.model_path}")
        else:
            print(f"[MLP] ⚠ Model not found at {self.model_path}. "
                  "Run train_mlp.py first.")

    def predict(self, feature_vector):
        """
        Predict gesture from a 42D feature vector.

        Args:
            feature_vector: np.ndarray of shape (42,)

        Returns:
            (label: str, confidence: float)
        """
        if not self.loaded:
            return ("HOVER", 0.0)

        with torch.no_grad():
            x = torch.from_numpy(feature_vector).float().unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            return (get_label(idx.item()), conf.item())
