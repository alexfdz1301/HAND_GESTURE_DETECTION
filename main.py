"""
main.py — ATLAS: Autonomous Terrain-Learning and Analysis System
==================================================================
MODE-1: Human-in-the-Loop Interaction

Two windows:
    1. Pygame — Isometric 3D drone simulator with telemetry HUD
    2. OpenCV — Real-time hand gesture camera feed (via gesture_process)

Usage:
    python3 main.py
    python3 main.py --no-gesture
    python3 main.py --no-gui
"""

import sys, os, math, time, argparse, multiprocessing, queue

# Ensure project root is on sys.path for local package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ───── ATLAS modules ─────
from drone.virtual_drone import VirtualDrone
from motion.motion_controller import MotionController
from motion.motion_commands import MotionCommand
from core.mode_manager import ModeManager

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ───── constants ─────
WIN_W, WIN_H = 960, 720
FPS = 30
CONF_THRESH = 0.6


# ══════════════════════════════════════════════════════════════
#  ISOMETRIC 3-D DRONE RENDERER  (Pygame)
# ══════════════════════════════════════════════════════════════

class DroneRenderer:
    """Pseudo-3D isometric drone visualiser."""

    # palette
    BG = (14, 16, 30)
    GRID  = (30, 34, 52)
    GRID_AXIS = (50, 56, 80)
    FLOOR = (20, 22, 38)
    BODY  = (30, 180, 240)
    ARM   = (80, 92, 115)
    ROTOR = (0, 255, 200)
    NOSE  = (255, 80, 80)
    ACCENT = (0, 200, 255)
    TXT   = (190, 195, 210)
    OK    = (80, 255, 130)
    WARN  = (255, 70, 70)
    EMO   = (255, 200, 50)

    ISO_X = math.cos(math.radians(30))   # ≈ 0.866
    ISO_Y = math.sin(math.radians(30))   # 0.5

    def __init__(self):
        pygame.init()
        self.w, self.h = WIN_W, WIN_H
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("ATLAS — 3-D Drone Simulator")
        self.clock = pygame.time.Clock()
        self.fn  = pygame.font.SysFont("monospace", 13)
        self.fm  = pygame.font.SysFont("monospace", 15, bold=True)
        self.fl  = pygame.font.SysFont("monospace", 19, bold=True)
        self.ft  = pygame.font.SysFont("monospace", 22, bold=True)
        self.tick = 0
        self.last_gesture = ""
        self.last_conf = 0.0

    # ---- coordinate helpers ----
    def iso(self, wx, wy, wz=0):
        """World (x,y,z) → screen pixel using isometric projection."""
        sx = self.w // 2 + int(wx * self.ISO_X - wy * self.ISO_X)
        sy = self.h // 2 + int(wx * self.ISO_Y + wy * self.ISO_Y) - int(wz)
        return sx, sy

    # ---- floor grid ----
    def draw_floor(self):
        span = 250
        step = 50
        for i in range(-span, span + 1, step):
            s = self.iso(i, -span)
            e = self.iso(i,  span)
            c = self.GRID_AXIS if i == 0 else self.GRID
            pygame.draw.line(self.screen, c, s, e, 1)
            s = self.iso(-span, i)
            e = self.iso( span, i)
            pygame.draw.line(self.screen, c, s, e, 1)

    # ---- drone ----
    def draw_drone(self, drone):
        self.tick += 1
        alt_px = drone.altitude * 1.5        # visual altitude scale
        cx, cy = self.iso(drone.x, drone.y, alt_px)

        base = 22 + int(drone.altitude / 18)
        arm  = int(base * 1.5)
        rr   = int(base * 0.52)
        br   = int(base * 0.38)
        hrad = math.radians(drone.heading - 90)

        # shadow on floor
        sx0, sy0 = self.iso(drone.x, drone.y, 0)
        a = max(25, 80 - int(drone.altitude / 4))
        ss = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.ellipse(ss, (0, 0, 0, a),
                            (sx0 - arm, sy0 - arm // 2, arm * 2, arm))
        self.screen.blit(ss, (0, 0))

        # altitude line (dashed)
        if drone.altitude > 0:
            for yy in range(0, int(alt_px), 6):
                s = self.iso(drone.x, drone.y, yy)
                e = self.iso(drone.x, drone.y, yy + 3)
                pygame.draw.line(self.screen, (60, 65, 90), s, e, 1)

        # rotor offsets
        offsets = [(arm, arm), (arm, -arm), (-arm, arm), (-arm, -arm)]
        cos_h, sin_h = math.cos(hrad), math.sin(hrad)
        rps = []
        for ox, oy in offsets:
            rx = int(cx + ox * cos_h - oy * sin_h)
            ry = int(cy + (ox * sin_h + oy * cos_h) * 0.5)  # squash y for iso
            rps.append((rx, ry))

        # arms
        for rp in rps:
            pygame.draw.line(self.screen, self.ARM, (cx, cy), rp, 3)

        # body
        if drone.is_emergency_stopped:
            p = int(abs(math.sin(self.tick * 0.15)) * 200)
            bc = (200 + p // 4, 30, 30)
        else:
            bc = self.BODY
        pygame.draw.circle(self.screen, bc, (cx, cy), br)
        pygame.draw.circle(self.screen, self.ACCENT, (cx, cy), br, 2)

        # rotors
        spd = 12 if drone.speed_mode == "fast" else 6
        for i, rp in enumerate(rps):
            rs = pygame.Surface((rr * 2 + 4, rr * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(rs, (0, 255, 200, 45), (rr + 2, rr + 2), rr)
            self.screen.blit(rs, (rp[0] - rr - 2, rp[1] - rr - 2))
            ang = (self.tick * spd + i * 90) % 360
            for b in range(2):
                a = math.radians(ang + b * 180)
                bx = int(rp[0] + rr * 0.88 * math.cos(a))
                by = int(rp[1] + rr * 0.88 * math.sin(a))
                pygame.draw.line(self.screen, self.ROTOR, rp, (bx, by), 2)
            pygame.draw.circle(self.screen, self.ROTOR, rp, 3)

        # heading nose
        nd = br + 7
        nx = int(cx + nd * math.cos(hrad))
        ny = int(cy + nd * math.sin(hrad) * 0.5)
        pygame.draw.circle(self.screen, self.NOSE, (nx, ny), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 2)

    # ---- HUD ----
    def draw_hud(self, drone, mm):
        s = drone.get_status()
        f = mm.get_status()

        # title
        self.screen.blit(self.ft.render(
            "ATLAS MODE-1  |  3D Simulator", True, self.ACCENT), (14, 10))
        pygame.draw.line(self.screen, self.ACCENT,
                         (14, 36), (self.w - 14, 36), 1)

        rows = [
            ("Pos", f"({s['x']:.0f}, {s['y']:.0f})"),
            ("Alt", f"{s['altitude']:.0f}"),
            ("Hdg", f"{s['heading']:.0f}°"),
            ("Spd", s['speed_mode']),
            ("", ""),
            ("State", f['state']),
            ("Cmd",   f['last_action']),
            ("Src",   f['last_source']),
        ]
        y = 45
        for lbl, val in rows:
            if not lbl:
                y += 6; continue
            self.screen.blit(self.fm.render(f"{lbl}:", True, (110, 115, 135)),
                             (16, y))
            vc = self.TXT
            if "emergency" in val.lower():
                vc = self.WARN
            self.screen.blit(self.fm.render(f" {val}", True, vc), (70, y))
            y += 20

        # gesture readout (right side)
        if self.last_gesture:
            gx = self.w - 260
            self.screen.blit(self.fm.render("Gesture:", True, (110, 115, 135)),
                             (gx, 48))
            self.screen.blit(self.fl.render(
                f" {self.last_gesture.upper()}", True, self.OK),
                (gx + 80, 46))
            # bar
            bx, by, bw, bh = gx, 72, 210, 10
            pygame.draw.rect(self.screen, (40, 44, 60), (bx, by, bw, bh), border_radius=4)
            fw = int(bw * self.last_conf)
            clr = (0, 200, 100) if self.last_conf > 0.8 else (
                  (200, 200, 0) if self.last_conf > 0.5 else (200, 80, 80))
            pygame.draw.rect(self.screen, clr, (bx, by, fw, bh), border_radius=4)
            self.screen.blit(self.fn.render(f"{self.last_conf:.0%}", True, self.TXT),
                             (bx + bw + 6, by - 2))

        # emergency banner
        if s["emergency_stopped"]:
            txt = self.ft.render("⛔  EMERGENCY STOP  ⛔", True, self.WARN)
            tw = txt.get_width()
            p = int(abs(math.sin(self.tick * .1)) * 80) + 40
            bg = pygame.Surface((tw + 30, 38), pygame.SRCALPHA)
            bg.fill((180, 0, 0, p))
            self.screen.blit(bg, (self.w // 2 - tw // 2 - 15, self.h - 52))
            self.screen.blit(txt, (self.w // 2 - tw // 2, self.h - 48))

    def draw_help(self):
        lines = [
            "WASD=move  QE=rot  ↑↓=alt  SPACE=hover  F=flip",
            "1=slow 2=fast 3=normal  X=e-stop  R=reset  ESC=quit",
        ]
        y = self.h - 48
        for l in lines:
            self.screen.blit(self.fn.render(l, True, (65, 68, 88)), (14, y))
            y += 16

    def render(self, drone, mm):
        self.screen.fill(self.BG)
        self.draw_floor()
        self.draw_drone(drone)
        self.draw_hud(drone, mm)
        self.draw_help()
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()


# ══════════════════════════════════════════════════════════════
#  KEYBOARD HANDLER
# ══════════════════════════════════════════════════════════════

def handle_keys(mm):
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            return False
        if ev.type != pygame.KEYDOWN:
            continue
        k = ev.key
        m = {
            pygame.K_w:     MotionCommand.MOVE_FORWARD,
            pygame.K_s:     MotionCommand.MOVE_BACKWARD,
            pygame.K_a:     MotionCommand.MOVE_LEFT,
            pygame.K_d:     MotionCommand.MOVE_RIGHT,
            pygame.K_UP:    MotionCommand.MOVE_UP,
            pygame.K_DOWN:  MotionCommand.MOVE_DOWN,
            pygame.K_q:     MotionCommand.ROTATE_LEFT,
            pygame.K_e:     MotionCommand.ROTATE_RIGHT,
            pygame.K_SPACE: MotionCommand.HOVER,
            pygame.K_f:     MotionCommand.FLIP,
            pygame.K_1:     MotionCommand.SLOW_MODE,
            pygame.K_2:     MotionCommand.FAST_MODE,
            pygame.K_3:     MotionCommand.NORMAL_MODE,
            pygame.K_x:     MotionCommand.EMERGENCY_STOP,
        }
        if k in m:
            mm.handle_keyboard(m[k])
        elif k == pygame.K_r:
            mm.reset_emergency()
        elif k == pygame.K_ESCAPE:
            return False
    return True


# ══════════════════════════════════════════════════════════════
#  GESTURE QUEUE POLLING
# ══════════════════════════════════════════════════════════════

def poll_gesture_queue(result_queue, mm, renderer):
    """Drain the gesture queue and route predictions to the FSM + HUD."""
    if result_queue is None:
        return
    while True:
        try:
            label, conf = result_queue.get_nowait()
        except queue.Empty:
            break

        if conf >= CONF_THRESH:
            mm.handle_gesture(label, conf)

        if renderer is not None:
            renderer.last_gesture = label
            renderer.last_conf = conf


# ══════════════════════════════════════════════════════════════
#  TERMINAL MODE
# ══════════════════════════════════════════════════════════════

def terminal_mode(mm, stop, result_queue=None):
    print("\n" + "=" * 55)
    print("  ATLAS MODE-1 — Terminal Mode")
    print("  Type: 'status', 'quit'")
    print("=" * 55 + "\n")
    while not stop.is_set():
        try:
            cmd = input("ATLAS> ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("quit", "exit", "q"):
                stop.set()
                break
            if cmd.lower() == "status":
                print(f"  {mm.drone}")
                continue
            poll_gesture_queue(result_queue, mm, None)
            print(f"  {mm.drone}")
        except (EOFError, KeyboardInterrupt):
            stop.set()
            break


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="ATLAS Drone Simulator")
    ap.add_argument("--no-gesture", action="store_true")
    ap.add_argument("--no-gui",     action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 55)
    print("  🚁  ATLAS — MODE-1: Human-in-the-Loop")
    print("  Autonomous Terrain-Learning and Analysis System")
    print("=" * 55 + "\n")

    drone = VirtualDrone()
    mm = ModeManager(drone=drone,
                     motion_controller=MotionController(drone))
    stop = multiprocessing.Event()

    # renderer
    renderer = None
    if not args.no_gui and PYGAME_AVAILABLE:
        renderer = DroneRenderer()

    # gesture + window (separate process using gesture_process.run)
    gesture_queue = None
    gesture_proc = None
    if not args.no_gesture and CV2_AVAILABLE and MEDIAPIPE_AVAILABLE:
        try:
            from gesture.gesture_process import run as gesture_run
            gesture_queue = multiprocessing.Queue(maxsize=1)
            gesture_proc = multiprocessing.Process(
                target=gesture_run,
                args=(gesture_queue, stop),
                daemon=True,
            )
            gesture_proc.start()
            print("[main] 📷 Gesture process started.")
        except ImportError as e:
            print(f"[main] Gesture process import failed: {e}")
    elif not args.no_gesture:
        missing = []
        if not CV2_AVAILABLE:
            missing.append("cv2")
        if not MEDIAPIPE_AVAILABLE:
            missing.append("mediapipe")
        print(f"[main] Gesture unavailable (missing: {', '.join(missing)}).")

    # main loop
    if args.no_gui or not PYGAME_AVAILABLE:
        terminal_mode(mm, stop, gesture_queue)
    else:
        print("[main] 🎮 Pygame window opened. Gesture capture running.")
        print("[main] Press ESC or close window to quit.\n")
        while not stop.is_set():
            # Apply velocity each frame
            drone.tick()
            poll_gesture_queue(gesture_queue, mm, renderer)
            if not handle_keys(mm):
                break
            renderer.render(drone, mm)
        renderer.close()

    stop.set()
    if gesture_proc:
        gesture_proc.join(timeout=2)
    print("\n[ATLAS] Shutdown complete. Goodbye! 🚁\n")


if __name__ == "__main__":
    main()
