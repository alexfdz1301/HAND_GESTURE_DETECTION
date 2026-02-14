"""
config.py — ATLAS v1 Central Configuration
=============================================
Single source of truth for all constants.
"""

# ─── Gesture Labels (7 motion-only) ───
GESTURE_LABELS = [
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "HOVER",
]
NUM_CLASSES = len(GESTURE_LABELS)
LABEL_TO_INDEX = {lbl: i for i, lbl in enumerate(GESTURE_LABELS)}
INDEX_TO_LABEL = {i: lbl for i, lbl in enumerate(GESTURE_LABELS)}

# ─── MLP Model ───
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 2   # 42D: (x, y) per landmark
MLP_HIDDEN_1 = 64
MLP_HIDDEN_2 = 32
MLP_DROPOUT = 0.2
MODEL_PATH = "models/gesture_mlp.pth"

# ─── Perception ───
CONF_THRESH = 0.6
EMA_ALPHA = 0.4
QUEUE_SIZE = 1
HAND_MARGIN = 30       # px padding around detected hand bbox

# ─── Renderer ───
WIN_W, WIN_H = 960, 720
FPS = 30

# ─── Drone ───
STEP_NORMAL = 10
MIN_POS, MAX_POS = -500, 500
MIN_ALT, MAX_ALT = 0, 200
