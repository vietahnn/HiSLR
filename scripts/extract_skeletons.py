#!/usr/bin/env python3
# scripts/extract_skeletons.py
# ─────────────────────────────────────────────────────────────────────────────
# Extract MediaPipe Holistic keypoints from every video in a directory.
#
# Produces one .npy file per video: shape (T, 75, 4)
#   Joints 0–32  : 33 body pose landmarks  (x, y, z, visibility)
#   Joints 33–53 : 21 right hand landmarks (x, y, z, visibility)
#   Joints 54–74 : 21 left  hand landmarks (x, y, z, visibility)
#
# Usage:
#   python scripts/extract_skeletons.py \
#       --video_dir  ./data/wlasl/videos \
#       --output_dir ./data/wlasl/skeletons \
#       --num_workers 4
#
# Dependencies: mediapipe>=0.10, opencv-python, tqdm
# ─────────────────────────────────────────────────────────────────────────────

import os
import argparse
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


NUM_BODY_JOINTS = 33
NUM_HAND_JOINTS = 21
NUM_TOTAL_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS * 2   # 75


def extract_one(video_path: str, output_path: str) -> bool:
    """
    Extract Holistic keypoints from a single video file.
    Saves a (T, 75, 4) float32 numpy array to output_path.

    Returns True on success, False on failure.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise ImportError("mediapipe and opencv-python are required. "
                          "pip install mediapipe opencv-python")

    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    frames_joints = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            joints = np.zeros((NUM_TOTAL_JOINTS, 4), dtype=np.float32)

            # ── Body pose (0–32) ─────────────────────────────────────────────
            if results.pose_landmarks:
                for j, lm in enumerate(results.pose_landmarks.landmark):
                    joints[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # ── Right hand (33–53) ───────────────────────────────────────────
            if results.right_hand_landmarks:
                for j, lm in enumerate(results.right_hand_landmarks.landmark):
                    joints[NUM_BODY_JOINTS + j] = [lm.x, lm.y, lm.z, 1.0]

            # ── Left hand (54–74) ────────────────────────────────────────────
            if results.left_hand_landmarks:
                for j, lm in enumerate(results.left_hand_landmarks.landmark):
                    joints[NUM_BODY_JOINTS + NUM_HAND_JOINTS + j] = [lm.x, lm.y, lm.z, 1.0]

            frames_joints.append(joints)

    cap.release()

    if len(frames_joints) == 0:
        return False

    joints_array = np.stack(frames_joints, axis=0)  # (T, 75, 4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, joints_array)
    return True


def _worker(args):
    video_path, output_path = args
    return video_path, extract_one(video_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic keypoints from videos")
    parser.add_argument("--video_dir",   type=str, required=True, help="Root directory of video files")
    parser.add_argument("--output_dir",  type=str, required=True, help="Output directory for .npy files")
    parser.add_argument("--extensions",  type=str, default="mp4,avi,mov", help="Video file extensions (comma-separated)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--overwrite",   action="store_true", help="Re-extract even if .npy already exists")
    args = parser.parse_args()

    extensions = [f".{e.strip().lstrip('.')}" for e in args.extensions.split(",")]

    # Collect all video paths
    video_paths = []
    for ext in extensions:
        video_paths.extend(glob.glob(os.path.join(args.video_dir, "**", f"*{ext}"), recursive=True))
    video_paths = sorted(set(video_paths))

    print(f"Found {len(video_paths)} videos in {args.video_dir}")

    # Build (input, output) pairs — mirror directory structure
    tasks = []
    for vp in video_paths:
        rel = os.path.relpath(vp, args.video_dir)
        out = os.path.join(args.output_dir, os.path.splitext(rel)[0] + ".npy")
        if not args.overwrite and os.path.exists(out):
            continue
        tasks.append((vp, out))

    print(f"Extracting {len(tasks)} videos (skipping {len(video_paths) - len(tasks)} already done)")

    failed = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extracting"):
            video_path, success = future.result()
            if not success:
                failed.append(video_path)

    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        fail_log = os.path.join(args.output_dir, "failed_extractions.txt")
        with open(fail_log, "w") as f:
            f.write("\n".join(failed))
        print(f"Failed videos logged to {fail_log}")


if __name__ == "__main__":
    main()
