"""
mode_manager.py — Simplified 2-State FSM
==========================================
States: IDLE, MOVING
Gesture → MotionCommand → Drone. That's it.
"""

from enum import Enum
from motion.motion_commands import MotionCommand, gesture_to_motion
from motion.motion_controller import MotionController


class DroneState(Enum):
    IDLE = "idle"
    MOVING = "moving"


class ModeManager:
    """
    Routes gesture labels to motion commands.
    Two states: IDLE and MOVING.
    """

    def __init__(self, drone, motion_controller):
        self.drone = drone
        self.motion_controller = motion_controller
        self.state = DroneState.IDLE
        self.last_action = "none"
        self.last_source = "none"

    def handle_gesture(self, gesture_label, confidence):
        """
        Process a gesture prediction → execute motion.

        Args:
            gesture_label: str, e.g. "MOVE_LEFT"
            confidence:    float 0–1

        Returns:
            True if executed
        """
        cmd = gesture_to_motion(gesture_label)
        if cmd is None:
            return False

        if cmd == MotionCommand.HOVER:
            self.state = DroneState.IDLE
        else:
            self.state = DroneState.MOVING

        self.motion_controller.execute(cmd)
        self.last_action = gesture_label
        self.last_source = "gesture"
        return True

    def handle_keyboard(self, cmd):
        """Execute a motion command from keyboard."""
        if cmd == MotionCommand.HOVER:
            self.state = DroneState.IDLE
        elif cmd == MotionCommand.EMERGENCY_STOP:
            self.state = DroneState.IDLE
        else:
            self.state = DroneState.MOVING
        self.motion_controller.execute(cmd)
        self.last_action = cmd.value
        self.last_source = "keyboard"

    def reset_emergency(self):
        """Reset emergency stop state."""
        self.drone.reset_emergency()
        self.state = DroneState.IDLE
        self.last_action = "RESET_EMERGENCY"
        self.last_source = "keyboard"

    def get_status(self):
        return {
            "state": self.state.value,
            "last_action": self.last_action,
            "last_source": self.last_source,
        }
