#!/usr/bin/env python3
# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# HiSLR — Evaluation and Demo Inference
#
# Usage:
#   # Evaluate on WLASL2000 test split
#   python evaluate.py --checkpoint ./outputs/hislr_wlasl2000/checkpoints/best.pth \
#                      --dataset wlasl2000 --data_root ./data/wlasl --split test
#
#   # Demo: predict a single video
#   python evaluate.py --checkpoint ./outputs/.../best.pth \
#                      --demo --video_path ./sample.mp4 --skeleton_path ./sample.npy
# ─────────────────────────────────────────────────────────────────────────────

import os
import argparse
import json
import numpy as np
import torch

from hislr.configs.default import get_wlasl2000_config, get_msasl_config, get_autsl_config
from hislr.inference import HiSLRInference


def parse_args():
    parser = argparse.ArgumentParser(description="HiSLR Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset",    type=str, default="wlasl2000",
                        choices=["wlasl100", "wlasl300", "wlasl1000", "wlasl2000", "msasl1000", "autsl"])
    parser.add_argument("--data_root",  type=str, default="./data/wlasl")
    parser.add_argument("--split",      type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device",     type=str, default="cuda")

    # Demo mode
    parser.add_argument("--demo",           action="store_true", help="Run single video demo")
    parser.add_argument("--video_path",     type=str, default="")
    parser.add_argument("--skeleton_path",  type=str, default="")
    parser.add_argument("--top_k",         type=int, default=5)

    return parser.parse_args()


def load_class_names(data_root: str) -> list:
    ann_path = os.path.join(data_root, "train_annotations.json")
    try:
        with open(ann_path) as f:
            ann = json.load(f)
        return ann.get("class_names", [])
    except Exception:
        return []


def main():
    args = parse_args()

    # Config
    cfg_map = {
        "wlasl2000": get_wlasl2000_config,
        "msasl1000": get_msasl_config,
        "autsl": get_autsl_config,
    }
    cfg = cfg_map.get(args.dataset, get_wlasl2000_config)()
    cfg.data.data_root = args.data_root

    class_names = load_class_names(args.data_root)

    engine = HiSLRInference(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        class_names=class_names,
        device=args.device,
    )

    if args.demo:
        # ── Single-video demo ─────────────────────────────────────────────────
        if not args.video_path or not args.skeleton_path:
            raise ValueError("--video_path and --skeleton_path required for --demo")

        # Load frames (simple decord example)
        try:
            import decord
            decord.bridge.set_bridge("numpy")
            vr = decord.VideoReader(args.video_path)
            T = cfg.data.num_frames
            indices = np.linspace(0, len(vr) - 1, T).astype(int)
            frames = vr.get_batch(indices).asnumpy()   # (T, H, W, 3)
        except Exception as e:
            print(f"Warning: could not load video ({e}), using zeros for demo")
            H, W = cfg.data.frame_size, cfg.data.frame_size
            frames = np.zeros((cfg.data.num_frames, H, W, 3), dtype=np.uint8)

        joints = np.load(args.skeleton_path).astype(np.float32)

        result = engine.predict(frames, joints, top_k=args.top_k)

        print("\n" + "=" * 50)
        print(f"  Predicted: {result['predicted_sign']}  ({result['confidence']*100:.1f}%)")
        print("-" * 50)
        print("  Top predictions:")
        for i, (sign, prob) in enumerate(zip(result["top_k_signs"], result["top_k_probs"])):
            print(f"    {i+1}. {sign:30s}  {prob*100:.1f}%")
        print("=" * 50)

    else:
        # ── Full dataset evaluation ───────────────────────────────────────────
        ann_file = os.path.join(args.data_root, f"{args.split}_annotations.json")
        results = engine.evaluate_dataset(ann_file, batch_size=args.batch_size)

        print("\n" + "=" * 50)
        print(f"  HiSLR Evaluation Results")
        print(f"  Dataset : {args.dataset} / {args.split}")
        print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
        print("-" * 50)
        print(f"  P-I Top-1 : {results['pi_top1']:.2f}%")
        print(f"  P-I Top-5 : {results['pi_top5']:.2f}%")
        print("=" * 50)


if __name__ == "__main__":
    main()
