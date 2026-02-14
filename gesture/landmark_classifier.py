"""
landmark_classifier.py — Rule-Based Gesture Classifier
=========================================================
Classifies gestures directly from MediaPipe hand landmarks
using geometric rules. No training required.

Supported gestures:
    MOVE_UP       — Index finger pointing up, other fingers closed
    MOVE_DOWN     — Hand pointing down (fingers below wrist)
    MOVE_LEFT     — Hand pointing left (fingers extend left)
    MOVE_RIGHT    — Hand pointing right (fingers extend right)
    MOVE_FORWARD  — Open palm, fingers spread (push forward)
    MOVE_BACKWARD — Closed fist (pull back)
    HOVER         — Two fingers up (peace sign / stop)

MediaPipe landmark indices:
    0=WRIST
    4=THUMB_TIP, 8=INDEX_TIP, 12=MIDDLE_TIP, 16=RING_TIP, 20=PINKY_TIP
    3=THUMB_IP,  6=INDEX_PIP, 10=MIDDLE_PIP, 14=RING_PIP, 18=PINKY_PIP
    5=INDEX_MCP, 9=MIDDLE_MCP, 13=RING_MCP, 17=PINKY_MCP
"""

import math


def _dist(a, b):
    """Euclidean distance between two landmarks."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _is_finger_extended(lms, tip_idx, pip_idx, mcp_idx):
    """
    Check if a finger is extended by comparing tip-to-wrist distance
    vs pip-to-wrist distance. Extended fingers have tip further from wrist.
    """
    wrist = lms[0]
    tip_dist = _dist(lms[tip_idx], wrist)
    pip_dist = _dist(lms[pip_idx], wrist)
    return tip_dist > pip_dist * 1.05


def _is_thumb_extended(lms):
    """Check if thumb is extended (tip further from palm center than IP joint)."""
    # Compare thumb tip x-distance from palm center
    palm_cx = (lms[0].x + lms[9].x) / 2
    tip_dx = abs(lms[4].x - palm_cx)
    ip_dx = abs(lms[3].x - palm_cx)
    return tip_dx > ip_dx * 1.1


def _finger_states(lms):
    """
    Return dict of which fingers are extended.
    """
    return {
        "thumb":  _is_thumb_extended(lms),
        "index":  _is_finger_extended(lms, 8, 6, 5),
        "middle": _is_finger_extended(lms, 12, 10, 9),
        "ring":   _is_finger_extended(lms, 16, 14, 13),
        "pinky":  _is_finger_extended(lms, 20, 18, 17),
    }


def _pointing_direction(lms):
    """
    Get the primary direction the hand is pointing.
    Uses vector from wrist (0) to middle fingertip (12).
    Returns (dx, dy) normalised, where:
        +x = right, -x = left (in image coords)
        +y = down, -y = up (in image coords, y increases downward)
    """
    dx = lms[12].x - lms[0].x
    dy = lms[12].y - lms[0].y
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-6:
        return 0, 0
    return dx / mag, dy / mag


def classify(hand_landmarks):
    """
    Classify a hand gesture from 21 MediaPipe NormalizedLandmarks.

    Args:
        hand_landmarks: list of landmarks with .x, .y attributes

    Returns:
        (gesture_label: str, confidence: float)
    """
    lms = hand_landmarks
    fingers = _finger_states(lms)
    n_extended = sum(fingers.values())
    dx, dy = _pointing_direction(lms)

    # Direction angles
    angle = math.degrees(math.atan2(dy, dx))  # -180 to 180
    # angle: 0=right, 90=down, -90=up, 180/-180=left

    # ── CLOSED FIST → MOVE_BACKWARD ──
    if n_extended <= 1 and not fingers["index"] and not fingers["middle"]:
        return ("MOVE_BACKWARD", 0.85)

    # ── PEACE SIGN (index + middle only) → HOVER ──
    if (fingers["index"] and fingers["middle"]
            and not fingers["ring"] and not fingers["pinky"]
            and n_extended <= 3):
        return ("HOVER", 0.90)

    # ── SINGLE INDEX FINGER → directional pointing ──
    if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        # Pointing up
        if angle < -45 and angle > -135:
            return ("MOVE_UP", 0.85)
        # Pointing down
        if angle > 45 and angle < 135:
            return ("MOVE_DOWN", 0.85)
        # Pointing left
        if abs(angle) > 135:
            return ("MOVE_LEFT", 0.85)
        # Pointing right
        if angle > -45 and angle < 45:
            return ("MOVE_RIGHT", 0.85)

    # ── OPEN PALM (all fingers out) → MOVE_FORWARD ──
    if n_extended >= 4:
        # Check if pointing up → MOVE_UP with open hand
        if dy < -0.6:
            return ("MOVE_UP", 0.80)
        # Check if pointing down → MOVE_DOWN with open hand
        if dy > 0.6:
            return ("MOVE_DOWN", 0.80)
        # Otherwise open palm = push forward
        return ("MOVE_FORWARD", 0.82)

    # ── THREE FINGERS → directional ──
    if n_extended == 3:
        if dy < -0.5:
            return ("MOVE_UP", 0.75)
        if dy > 0.5:
            return ("MOVE_DOWN", 0.75)
        if dx < -0.5:
            return ("MOVE_LEFT", 0.75)
        if dx > 0.5:
            return ("MOVE_RIGHT", 0.75)

    # ── DEFAULT → HOVER ──
    return ("HOVER", 0.60)
