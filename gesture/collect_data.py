"""
collect_data.py — Gesture Dataset Collection Tool
====================================================
Uses a webcam + MediaPipe Hands to capture hand gesture images.
Saves cropped, resized (128×128) grayscale images into a folder-per-class
structure suitable for training the GestureCNN.

Usage:
    python -m gesture.collect_data --label thumb_up --split train --count 800
    python -m gesture.collect_data --label thumb_up --split val   --count 200

Dataset structure created:
    dataset/
      train/
        up/         (800 images)
        down/       (800 images)
        ...
      val/
        up/         (200 images)
        down/       (200 images)
        ...
"""

import os
import sys
import cv2
import argparse
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils as mp_drawing,
)
from gesture.gesture_labels import GESTURE_LABELS


# ---------- configuration ----------

DATASET_ROOT = "dataset"
IMAGE_SIZE = 128       # resize hand crop to this square size
MARGIN = 30            # extra pixels around the detected hand

# Path to the hand landmarker model (downloaded from MediaPipe)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Hand connections for drawing
_HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


def collect_images(label, split, count):
    """
    Open the webcam and collect hand gesture images for one class.

    Args:
        label: gesture class name (e.g. "thumb_up")
        split: "train" or "val"
        count: number of images to capture
    """
    # Validate label
    if label not in GESTURE_LABELS:
        print(f"[ERROR] '{label}' is not a valid gesture label.")
        print(f"  Valid labels: {GESTURE_LABELS}")
        sys.exit(1)

    # Create output directory
    save_dir = os.path.join(DATASET_ROOT, split, label)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing images to avoid overwriting
    existing = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
    print(f"\n=== Collecting '{label}' ({split}) ===")
    print(f"  Target: {count} images | Already have: {existing}")
    print(f"  Save to: {save_dir}")
    print("  Press 'c' to capture | 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    captured = 0

    # Set up the HandLandmarker (new Tasks API)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_timestamp_ms = 0

        while captured < count:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read from webcam.")
                break

            # Flip horizontally for natural mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe Image and detect
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms += 33  # ~30 fps
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            hand_crop = None

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # Draw hand landmarks on preview
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, _HAND_CONNECTIONS
                    )

                    # Calculate bounding box from landmarks
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in hand_landmarks]
                    y_coords = [lm.y * h for lm in hand_landmarks]

                    x_min = max(0, int(min(x_coords)) - MARGIN)
                    y_min = max(0, int(min(y_coords)) - MARGIN)
                    x_max = min(w, int(max(x_coords)) + MARGIN)
                    y_max = min(h, int(max(y_coords)) + MARGIN)

                    # Crop and resize
                    crop = frame[y_min:y_max, x_min:x_max]
                    if crop.size > 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        hand_crop = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))

                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                  (0, 255, 0), 2)

            # Show status
            status = (f"Label: {label} | Split: {split} | "
                      f"Captured: {captured}/{count}")
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Gesture Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and hand_crop is not None:
                # Save the image
                filename = f"{label}_{existing + captured:05d}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, hand_crop)
                captured += 1
                print(f"  Saved [{captured}/{count}]: {filename}")

            elif key == ord('q'):
                print("\n  Quit early by user.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Captured {captured} images for '{label}' ({split}).\n")


# ---------- CLI entry point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect hand gesture images for ATLAS CNN training."
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help=f"Gesture class to collect. Options: {GESTURE_LABELS}"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val"],
        help="Dataset split: 'train' or 'val' (default: train)"
    )
    parser.add_argument(
        "--count", type=int, default=800,
        help="Number of images to capture (default: 800)"
    )
    args = parser.parse_args()
    collect_images(args.label, args.split, args.count)
