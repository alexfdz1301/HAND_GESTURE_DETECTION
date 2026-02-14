"""
motion_controller.py — Motion Execution Controller
=====================================================
Dispatches MotionCommands to VirtualDrone.
"""

from motion.motion_commands import MotionCommand


class MotionController:

    def __init__(self, drone):
        self.drone = drone
        self._dispatch = {
            MotionCommand.MOVE_UP:       self.drone.move_up,
            MotionCommand.MOVE_DOWN:     self.drone.move_down,
            MotionCommand.MOVE_LEFT:     self.drone.move_left,
            MotionCommand.MOVE_RIGHT:    self.drone.move_right,
            MotionCommand.MOVE_FORWARD:  self.drone.move_forward,
            MotionCommand.MOVE_BACKWARD: self.drone.move_backward,
            MotionCommand.HOVER:         self.drone.hover,
            MotionCommand.ROTATE_LEFT:   lambda: self.drone.rotate(45),
            MotionCommand.ROTATE_RIGHT:  lambda: self.drone.rotate(-45),
            MotionCommand.FLIP:          self.drone.flip,
            MotionCommand.SLOW_MODE:     lambda: self.drone.set_speed_mode("slow"),
            MotionCommand.FAST_MODE:     lambda: self.drone.set_speed_mode("fast"),
            MotionCommand.NORMAL_MODE:   lambda: self.drone.set_speed_mode("normal"),
            MotionCommand.EMERGENCY_STOP: self.drone.emergency_stop,
        }

    def execute(self, command):
        """
        Execute a motion command. Returns True if executed.
        """
        action = self._dispatch.get(command)
        if action is None:
            return False
        action()
        return True
