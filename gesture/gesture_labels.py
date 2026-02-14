"""
gesture_labels.py — 7 Motion-Only Gesture Classes
====================================================
"""

from config import GESTURE_LABELS, NUM_CLASSES, LABEL_TO_INDEX, INDEX_TO_LABEL


def get_label(index):
    """Return gesture label for a given class index, or 'unknown'."""
    return INDEX_TO_LABEL.get(index, "unknown")


def get_index(label):
    """Return class index for a given gesture label, or -1."""
    return LABEL_TO_INDEX.get(label, -1)
