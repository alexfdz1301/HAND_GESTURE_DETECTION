"""
motion_commands.py — Motion Commands
======================================
All motion commands supported by the drone.
"""

from enum import Enum


class MotionCommand(Enum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    HOVER = "HOVER"
    ROTATE_LEFT = "ROTATE_LEFT"
    ROTATE_RIGHT = "ROTATE_RIGHT"
    FLIP = "FLIP"
    SLOW_MODE = "SLOW_MODE"
    FAST_MODE = "FAST_MODE"
    NORMAL_MODE = "NORMAL_MODE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


# Gesture label (str) → MotionCommand
# Only the 7 core motion commands are mapped from gestures
GESTURE_TO_MOTION = {
    "MOVE_UP":       MotionCommand.MOVE_UP,
    "MOVE_DOWN":     MotionCommand.MOVE_DOWN,
    "MOVE_LEFT":     MotionCommand.MOVE_LEFT,
    "MOVE_RIGHT":    MotionCommand.MOVE_RIGHT,
    "MOVE_FORWARD":  MotionCommand.MOVE_FORWARD,
    "MOVE_BACKWARD": MotionCommand.MOVE_BACKWARD,
    "HOVER":         MotionCommand.HOVER,
}


def gesture_to_motion(gesture_label):
    """Convert a gesture label string to a MotionCommand, or None."""
    return GESTURE_TO_MOTION.get(gesture_label, None)
