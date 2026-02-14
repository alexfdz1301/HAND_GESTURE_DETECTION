"""
virtual_drone.py — Virtual Drone Simulation
==============================================
Continuous velocity-based motion model.
Gesture sets velocity, HOVER zeroes it.
Drone position updates every tick.
"""

import time
from config import STEP_NORMAL, MIN_POS, MAX_POS, MIN_ALT, MAX_ALT


class VirtualDrone:
    """Virtual drone with position (x, y), altitude, and velocity."""

    SPEED_SLOW = STEP_NORMAL * 0.5
    SPEED_NORMAL = STEP_NORMAL
    SPEED_FAST = STEP_NORMAL * 2.0

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.altitude = 50.0   # start airborne
        self.heading = 0.0

        # Velocity components (set by gesture, zeroed by HOVER)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.speed = self.SPEED_NORMAL
        self.speed_mode = "normal"
        self.is_emergency_stopped = False
        self.last_action = "HOVER"
        self.state_log = []

    # ── velocity setters (called by motion commands) ──

    def move_up(self):
        if self.is_emergency_stopped:
            return
        self.vx, self.vy, self.vz = 0, 0, self.speed
        self._log("MOVE_UP")

    def move_down(self):
        if self.is_emergency_stopped:
            return
        self.vx, self.vy, self.vz = 0, 0, -self.speed
        self._log("MOVE_DOWN")

    def move_left(self):
        if self.is_emergency_stopped:
            return
        self.vx, self.vy, self.vz = -self.speed, 0, 0
        self._log("MOVE_LEFT")

    def move_right(self):
        if self.is_emergency_stopped:
            return
        self.vx, self.vy, self.vz = self.speed, 0, 0
        self._log("MOVE_RIGHT")

    def move_forward(self):
        if self.is_emergency_stopped:
            return
        self.vx, self.vy, self.vz = 0, self.speed, 0
        self._log("MOVE_FORWARD")

    def move_backward(self):
        if self.is_emergency_stopped:
            return
        self.vx, self.vy, self.vz = 0, -self.speed, 0
        self._log("MOVE_BACKWARD")

    def hover(self):
        self.vx, self.vy, self.vz = 0, 0, 0
        self._log("HOVER")

    def rotate(self, degrees=45):
        if self.is_emergency_stopped:
            return
        self.heading = (self.heading + degrees) % 360
        self._log(f"ROTATE({degrees})")

    def flip(self):
        if self.is_emergency_stopped:
            return
        self._log("FLIP")

    def set_speed_mode(self, mode):
        if mode == "slow":
            self.speed = self.SPEED_SLOW
            self.speed_mode = "slow"
        elif mode == "fast":
            self.speed = self.SPEED_FAST
            self.speed_mode = "fast"
        else:
            self.speed = self.SPEED_NORMAL
            self.speed_mode = "normal"
        self._log(f"SPEED_{mode.upper()}")

    def emergency_stop(self):
        self.is_emergency_stopped = True
        self.vx, self.vy, self.vz = 0, 0, 0
        self._log("EMERGENCY_STOP")

    def reset_emergency(self):
        self.is_emergency_stopped = False
        self._log("RESET_EMERGENCY")

    # ── tick: apply velocity ──

    def tick(self):
        """Update position based on current velocity. Call once per frame."""
        if self.is_emergency_stopped:
            return
        self.x = self._clamp(self.x + self.vx, MIN_POS, MAX_POS)
        self.y = self._clamp(self.y + self.vy, MIN_POS, MAX_POS)
        self.altitude = self._clamp(self.altitude + self.vz, MIN_ALT, MAX_ALT)

    # ── helpers ──

    @staticmethod
    def _clamp(val, lo, hi):
        return max(lo, min(hi, val))

    def _log(self, action):
        self.last_action = action
        self.state_log.append({
            "action": action, "time": time.time(),
            "pos": (self.x, self.y, self.altitude),
        })
        if len(self.state_log) > 100:
            self.state_log = self.state_log[-50:]

    def get_status(self):
        return {
            "x": self.x,
            "y": self.y,
            "altitude": self.altitude,
            "heading": self.heading,
            "speed_mode": self.speed_mode,
            "emergency_stopped": self.is_emergency_stopped,
            "last_action": self.last_action,
            "vx": self.vx, "vy": self.vy, "vz": self.vz,
        }

    def __repr__(self):
        return (f"VirtualDrone(x={self.x:.0f}, y={self.y:.0f}, "
                f"alt={self.altitude:.0f}, v=({self.vx},{self.vy},{self.vz}))")
