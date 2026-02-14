# ATLAS – AIMS Gesture Controlled Drone System

ATLAS (AIMS Landmarked Autonomous System) is a real-time, gesture-controlled drone simulation framework built on computer vision, geometric feature engineering, and machine learning. A webcam + MediaPipe extract hand landmarks; a lightweight MLP classifies gestures and drives a virtual drone via a deterministic control stack.

## Current Status
- **Mode 1 – ML-Based Gesture Control** 
- **Mode 2 – Advanced Temporal Control** (architected, not yet active)

## System Architecture
ATLAS follows a layered perception → decision → control → simulation pipeline:

```
Webcam
  ↓
MediaPipe Hand Landmark Model (21 landmarks: x, y, z)
  ↓
Feature Engineering (wrist-centering, scale normalize → 42D vector)
  ↓
MLP Classifier (GestureMLP)
  ↓
Finite State Machine (IDLE / MOVING)
  ↓
MotionController (MotionCommand enum)
  ↓
Virtual Drone (velocity-based physics)
  ↓
Pygame Renderer
```

This modular separation keeps perception, decision, control, and simulation cleanly decoupled for scalability and real-time performance.

## Modes
### Mode 1 – ML-Based Gesture Control (Active)
- Live camera → MediaPipe → 42D normalized feature vector
- MLP predicts gesture class
- FSM validates state to prevent flicker
- MotionCommand maps to velocity
- Virtual drone updates; Pygame renders

### Mode 2 – Advanced Temporal Control (Planned)
- Temporal modeling (e.g., LSTM / sequence models)
- Confidence-based thresholding and debouncing
- Expanded gesture vocabulary
- Reduced reliance on FSM smoothing
- Hardware drone integration

## Gesture Classes (Mode 1)
| Gesture        | Drone Action      |
|:---------------|:------------------|
| `MOVE_UP`      | Gain altitude     |
| `MOVE_DOWN`    | Lose altitude     |
| `MOVE_LEFT`    | Strafe left       |
| `MOVE_RIGHT`   | Strafe right      |
| `MOVE_FORWARD` | Move forward      |
| `MOVE_BACKWARD`| Move backward     |
| `HOVER`        | Stop (zero velocity) |

## Machine Learning Implementation
**Feature Engineering**
- Input: 21 hand landmarks → 42D vector (x, y)
- Translation normalization: wrist-centered
- Scale normalization: relative hand-size scaling
- Embeds translation/scale invariance into representation (reduces variance, improves generalization).

**Model Architecture**
- Type: MLP (GestureMLP)
- Input: 42
- Hidden layers: FC + ReLU
- Output: multiclass logits (7 gestures)
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Weights: `models/gesture_mlp.pth`
- Inference is lightweight for low-latency control.

## Control Logic
**Finite State Machine (FSM)**
- States: `IDLE`, `MOVING`
- Purpose: dampen prediction flicker, stabilize motion, enforce deterministic transitions.

**Motion Abstraction Layer**
- `MotionCommand` enum + `MotionController` dispatcher decouple ML outputs from physics.

**Drone Simulation**
- Velocity-based integration (smooth, continuous updates per frame).

## Project Structure
```
ATLAS-Aims-drone/
├── config.py                  # Central constants
├── main.py                    # Pygame renderer + main loop
├── drone/
│   └── virtual_drone.py       # Velocity-based drone model
├── motion/
│   ├── motion_commands.py     # MotionCommand enum
│   └── motion_controller.py   # Command → drone dispatch
├── core/
│   └── mode_manager.py        # FSM (IDLE / MOVING)
├── gesture/
│   ├── gesture_process.py     # Perception process (camera + MLP)
│   ├── feature_engineering.py # Landmarks → 42D vector
│   ├── mlp_model.py           # GestureMLP + GestureClassifier
│   ├── train_mlp.py           # Training script
│   ├── gesture_labels.py      # Label utilities
│   └── hand_landmarker.task   # MediaPipe model file
├── models/
│   └── gesture_mlp.pth        # Trained MLP weights
└── dataset/
    └── landmarks_train.npy    # Training data
```

## Installation & Setup
```bash
# 1) Clone
git clone <your-repo-link>
cd ATLAS-Aims-drone

# 2) (Recommended) Virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 3) Install dependencies
pip install torch torchvision numpy mediapipe pygame

# 4) Run Mode 1 (gesture control; requires webcam)
python main.py

# Keyboard-only fallback
python main.py --no-gesture
```

## Controls
**Keyboard**
- WASD — move (fwd/back/left/right)
- ↑ / ↓ — altitude up/down
- SPACE — hover (stop)
- ESC — quit

**Gesture**
- Perform any of the 7 gestures above; the classifier publishes `(label, confidence)` to the main process queue, which maps to `MotionCommand`.

## Configuration
- **`config.py`**: Adjust smoothing (EMA), confidence thresholds, speeds, window sizes, and other tuning constants.
- **Gesture pipeline**: `gesture/gesture_process.py` manages MediaPipe, feature extraction, smoothing, and send-on-change.
- **Control loop**: `main.py` wires `ModeManager` → `MotionController` → `VirtualDrone` → renderer.

## Dataset & Training
- Dataset: `dataset/landmarks_train.npy` (42D feature vectors + labels).
- Training script: `gesture/train_mlp.py`
  ```bash
  python gesture/train_mlp.py --epochs 50 --batch-size 128 --lr 1e-3
  ```
- Replace `models/gesture_mlp.pth` with new weights after retraining.

## Demo
- YouTube Demo: *<Insert Unlisted YouTube Link Here>*
- Google Drive (project files): *<Insert Drive Link Here>*

## Tips & Troubleshooting
- Increase EMA smoothing or confidence thresholds in `config.py` if gestures feel jittery.
- Use `--no-gesture` to debug rendering/control independently of perception.
- Good lighting and minimal occlusion greatly improve MediaPipe landmark quality.
- Ensure `gesture/hand_landmarker.task` and `models/gesture_mlp.pth` exist before running gesture mode.

## Technical Highlights
- Landmark-based geometric feature engineering
- Translation & scale invariant preprocessing
- Lightweight MLP optimized for latency
- ML integrated with deterministic FSM-based control
- Modular, decoupled architecture
- Real-time velocity-based simulation

## Future Improvements
- Temporal sequence modeling (LSTM / Temporal CNN)
- Confidence-based debouncing and per-gesture cooldowns
- Dropout & BatchNorm for robustness
- Model quantization for embedded deployment
- Expanded gesture vocabulary
- Hardware drone integration

## Dependencies
- Python 3.x
- PyTorch
- MediaPipe
- NumPy
- Pygame

## Contributing
Issues and PRs are welcome. Please keep changes modular (perception, control, rendering) and document any tuning updates in `config.py`.

