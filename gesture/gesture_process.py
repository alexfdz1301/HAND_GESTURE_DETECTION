"""
gesture_process.py — Perception Process (Separate Process)
=============================================================
Owns the webcam + OpenCV window.
Pipeline:  Camera → MediaPipe → Landmark Classifier → Queue

Uses rule-based landmark classification (no training required).
Falls back to MLP if trained model exists.

Queue discipline:
    - maxsize=1, overwrite old value
    - Only send on gesture CHANGE with conf ≥ threshold
    - On hand lost (debounced) → send single HOVER event
    - Never spam HOVER continuously
"""

import os
import sys
import queue

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    HandLandmarksConnections, RunningMode,
    drawing_utils as mp_drawing,
)

# Ensure project root on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONF_THRESH, EMA_ALPHA, HAND_MARGIN
from gesture.landmark_classifier import classify as landmark_classify

HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
_MODEL_TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "hand_landmarker.task")

# Debounce: number of consecutive no-hand frames before declaring "hand lost"
HAND_LOST_THRESHOLD = 15


def run(result_queue, stop_event, model_path=None):
    """
    Main perception loop. Meant to be called via multiprocessing.Process.

    Args:
        result_queue:  multiprocessing.Queue(maxsize=1)
        stop_event:    multiprocessing.Event
        model_path:    (unused, kept for API compat)
    """
    # ── open webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Gesture] ❌ Cannot open webcam.")
        return

    # ── MediaPipe hand landmarker ──
    if not os.path.exists(_MODEL_TASK_PATH):
        print(f"[Gesture] ❌ hand_landmarker.task not found at {_MODEL_TASK_PATH}")
        cap.release()
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_TASK_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        print("[Gesture] ✅ MediaPipe hand landmarker loaded.")
        print("[Gesture] 📷 Gesture window opened.")
        print("[Gesture] Using rule-based landmark classifier.")

        frame_ts = 0
        prev_gesture = None
        hand_missing_count = 0
        hover_sent = False        # prevents repeated HOVER sends

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ── detect landmarks ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33
            try:
                results = landmarker.detect_for_video(mp_image, frame_ts)
            except Exception:
                results = None

            gesture_label = None
            gesture_conf = 0.0
            hand_detected = False

            if results and results.hand_landmarks:
                for hand_lms in results.hand_landmarks:
                    hand_detected = True
                    hand_missing_count = 0
                    hover_sent = False

                    # Draw landmarks on frame
                    mp_drawing.draw_landmarks(frame, hand_lms, HAND_CONNECTIONS)

                    # Bounding box
                    xs = [lm.x * w for lm in hand_lms]
                    ys = [lm.y * h for lm in hand_lms]
                    x1 = max(0, int(min(xs)) - HAND_MARGIN)
                    y1 = max(0, int(min(ys)) - HAND_MARGIN)
                    x2 = min(w, int(max(xs)) + HAND_MARGIN)
                    y2 = min(h, int(max(ys)) + HAND_MARGIN)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # ── Classify gesture from landmarks ──
                    gesture_label, gesture_conf = landmark_classify(hand_lms)

                    break  # first hand only

            # ── Queue logic ──
            if hand_detected and gesture_label is not None:
                if gesture_conf >= CONF_THRESH and gesture_label != prev_gesture:
                    _send(result_queue, gesture_label, gesture_conf)
                    print(f"[GESTURE] {gesture_label} ({gesture_conf:.2f})")
                    prev_gesture = gesture_label

            elif not hand_detected:
                hand_missing_count += 1
                # Only send HOVER once after debounce threshold
                if hand_missing_count >= HAND_LOST_THRESHOLD and not hover_sent:
                    _send(result_queue, "HOVER", 1.0)
                    print("[GESTURE] HOVER (hand lost)")
                    prev_gesture = "HOVER"
                    hover_sent = True

            # ── HUD overlay ──
            cv2.rectangle(frame, (0, 0), (w, 65), (12, 14, 28), -1)
            cv2.putText(frame, "ATLAS Gesture Window", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            if hand_detected and gesture_label:
                clr = (0, 255, 130) if gesture_conf >= CONF_THRESH else (0, 180, 255)
                cv2.putText(frame, f"{gesture_label}  {gesture_conf:.0%}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, clr, 2)

                # Show finger states for debugging
                from gesture.landmark_classifier import _finger_states
                fingers = _finger_states(hand_lms)
                finger_str = " ".join(
                    f.upper()[0] for f, v in fingers.items() if v
                )
                cv2.putText(frame, f"Fingers: {finger_str if finger_str else 'none'}",
                            (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (180, 180, 220), 1)
            else:
                cv2.putText(frame, "No hand detected", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 200), 2)

            cv2.imshow("ATLAS Gesture Window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[Gesture] Gesture window closed.")


def _send(q, label, conf):
    """Put into queue, discarding old value if full."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait((label, conf))
    except queue.Full:
        pass
