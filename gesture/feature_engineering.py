"""
feature_engineering.py — Landmark → Feature Vector
=====================================================
Converts 21 MediaPipe hand landmarks into a normalised 42D vector.

Pipeline:
    1. Extract (x, y) for each of the 21 landmarks
    2. Centre on wrist (landmark 0)
    3. Normalise by max Euclidean distance from wrist
    4. Flatten to a 42-element vector
"""

import numpy as np


def landmarks_to_feature(hand_landmarks):
    """
    Convert a list of 21 NormalizedLandmark objects to a 42D numpy vector.

    Args:
        hand_landmarks: list of landmarks with .x, .y attributes
                        (MediaPipe NormalizedLandmark format)

    Returns:
        np.ndarray of shape (42,), dtype float32
    """
    # Step 1: extract raw (x, y)
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks], dtype=np.float32)

    # Step 2: centre on wrist (landmark 0)
    wrist = coords[0].copy()
    coords -= wrist

    # Step 3: normalise by max distance from wrist
    distances = np.linalg.norm(coords, axis=1)
    max_dist = distances.max()
    if max_dist > 1e-6:
        coords /= max_dist

    # Step 4: flatten to 42D
    return coords.flatten()
