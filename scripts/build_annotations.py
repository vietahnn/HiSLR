#!/usr/bin/env python3
# scripts/build_annotations.py
# ─────────────────────────────────────────────────────────────────────────────
# Build HiSLR-format annotation JSON files from dataset split files.
# Supports: WLASL, MS-ASL, AUTSL
#
# Output format (one JSON per split: train/val/test):
# {
#     "num_classes": 2000,
#     "class_names": ["hello", "world", ...],
#     "samples": [
#         {
#             "video_path": "...",
#             "skeleton_path": "...",
#             "gloss_label": 42,
#             "phon_labels": [3, 12, -1, 7, ...]   // 16 ints, -1 = missing
#         }
#     ]
# }
#
# Usage (WLASL):
#   python scripts/build_annotations.py \
#       --dataset wlasl \
#       --video_dir ./data/wlasl/videos \
#       --skeleton_dir ./data/wlasl/skeletons \
#       --split_file ./data/wlasl/WLASL_v0.3.json \
#       --asl_lex ./data/asl_lex.json \
#       --output_dir ./data/wlasl \
#       --num_classes 2000
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional


# ── ASL-LEX phonological annotation loader ────────────────────────────────────

PHON_ATTRIBUTES = [
    "dominant_handshape",
    "nondominant_handshape",
    "major_location",
    "minor_location",
    "contact",
    "dominant_movement",
    "path_movement",
    "wrist_twist",
    "palm_orientation",
    "nmm_brows",
    "nmm_eyes",
    "nmm_cheeks",
    "nmm_mouth",
    "nmm_tongue",
    "nmm_head",
    "nmm_shoulders",
]

PHON_NUM_CLASSES = [87, 87, 9, 32, 4, 17, 6, 4, 6, 2, 2, 2, 2, 2, 2, 2]


def load_asl_lex(path: Optional[str]) -> Dict[str, List[int]]:
    """
    Load ASL-LEX 2.0 phonological annotations.
    Returns dict mapping gloss (lowercase) -> list of 16 integer labels.
    Missing values are -1.

    Expected JSON format (simplified):
    {
        "hello": {
            "dominant_handshape": 3,
            "major_location": 5,
            ...
        }
    }
    """
    if not path or not os.path.exists(path):
        print("[build_annotations] No ASL-LEX file found; phonological labels will all be -1")
        return {}

    with open(path) as f:
        raw = json.load(f)

    result = {}
    for gloss, attrs in raw.items():
        labels = []
        for attr in PHON_ATTRIBUTES:
            labels.append(attrs.get(attr, -1))
        result[gloss.lower().strip()] = labels

    print(f"[build_annotations] Loaded ASL-LEX for {len(result)} glosses")
    return result


# ── WLASL builder ──────────────────────────────────────────────────────────────

def build_wlasl(
    video_dir: str,
    skeleton_dir: str,
    split_file: str,
    output_dir: str,
    num_classes: int,
    asl_lex: dict,
):
    """
    Parse WLASL_v0.3.json and build train/val/test annotation files.
    """
    with open(split_file) as f:
        wlasl_data = json.load(f)  # list of {gloss, instances: [{video_id, split, ...}]}

    # Build vocabulary (sorted by frequency, capped at num_classes)
    gloss_counts = defaultdict(int)
    for entry in wlasl_data:
        for inst in entry["instances"]:
            gloss_counts[entry["gloss"]] += 1

    top_glosses = sorted(gloss_counts.keys(), key=lambda g: -gloss_counts[g])[:num_classes]
    gloss_to_idx = {g: i for i, g in enumerate(top_glosses)}

    splits = {"train": [], "val": [], "test": []}

    missing_videos = 0
    missing_skeletons = 0

    for entry in wlasl_data:
        gloss = entry["gloss"]
        if gloss not in gloss_to_idx:
            continue

        label = gloss_to_idx[gloss]
        phon_labels = asl_lex.get(gloss.lower(), [-1] * 16)

        for inst in entry["instances"]:
            video_id = inst["video_id"]
            split = inst.get("split", "train")

            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            skeleton_path = os.path.join(skeleton_dir, f"{video_id}.npy")

            if not os.path.exists(video_path):
                missing_videos += 1
                continue
            if not os.path.exists(skeleton_path):
                missing_skeletons += 1
                # Still include — skeleton will return zeros, degraded performance

            sample = {
                "video_path":    video_path,
                "skeleton_path": skeleton_path,
                "gloss_label":   label,
                "gloss":         gloss,
                "phon_labels":   phon_labels,
            }

            if split in splits:
                splits[split].append(sample)

    print(f"[WLASL] Missing videos: {missing_videos}, missing skeletons: {missing_skeletons}")

    os.makedirs(output_dir, exist_ok=True)
    for split_name, samples in splits.items():
        out = {
            "num_classes": num_classes,
            "class_names": top_glosses,
            "samples": samples,
        }
        out_path = os.path.join(output_dir, f"{split_name}_annotations.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[WLASL] {split_name}: {len(samples)} samples -> {out_path}")


# ── MS-ASL builder ─────────────────────────────────────────────────────────────

def build_msasl(
    video_dir: str,
    skeleton_dir: str,
    split_file: str,
    output_dir: str,
    num_classes: int,
    asl_lex: dict,
):
    """
    Parse MS-ASL annotation JSON.
    Expected format: list of {clean_text, file, label, ...} per split.
    MS-ASL provides separate train/val/test JSON files; split_file should
    point to a directory containing train.json, val.json, test.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    split_names = ["train", "val", "test"]

    # Collect all labels to build vocabulary
    all_labels = set()
    for sn in split_names:
        fp = os.path.join(split_file, f"{sn}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
            for item in data:
                all_labels.add(item.get("clean_text", ""))

    top_glosses = sorted(all_labels)[:num_classes]
    gloss_to_idx = {g: i for i, g in enumerate(top_glosses)}

    for sn in split_names:
        fp = os.path.join(split_file, f"{sn}.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)

        samples = []
        for item in data:
            gloss = item.get("clean_text", "")
            if gloss not in gloss_to_idx:
                continue

            video_id = os.path.splitext(os.path.basename(item.get("file", "")))[0]
            video_path    = os.path.join(video_dir, f"{video_id}.mp4")
            skeleton_path = os.path.join(skeleton_dir, f"{video_id}.npy")

            if not os.path.exists(video_path):
                continue

            samples.append({
                "video_path":    video_path,
                "skeleton_path": skeleton_path,
                "gloss_label":   gloss_to_idx[gloss],
                "gloss":         gloss,
                "phon_labels":   asl_lex.get(gloss.lower(), [-1] * 16),
            })

        out = {"num_classes": num_classes, "class_names": top_glosses, "samples": samples}
        out_path = os.path.join(output_dir, f"{sn}_annotations.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[MS-ASL] {sn}: {len(samples)} samples -> {out_path}")


# ── AUTSL builder ──────────────────────────────────────────────────────────────

def build_autsl(
    video_dir: str,
    skeleton_dir: str,
    split_file: str,
    output_dir: str,
    num_classes: int,
    asl_lex: dict,
):
    """
    Parse AUTSL annotation CSV files.
    Expected: split_file is a directory with train_labels.csv, val_labels.csv, test_labels.csv.
    CSV format: filename,label_id
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)
    class_names_path = os.path.join(split_file, "SignList_ClassId_TR_EN_v3.csv")
    class_names = {}

    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names[int(row["ClassId"])] = row.get("EN", row.get("TR", f"class_{row['ClassId']}"))

    top_glosses = [class_names.get(i, f"class_{i}") for i in range(num_classes)]

    for sn in ["train", "val", "test"]:
        csv_path = os.path.join(split_file, f"{sn}_labels.csv")
        if not os.path.exists(csv_path):
            continue

        samples = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                filename, label_id = row[0].strip(), int(row[1].strip())
                if label_id >= num_classes:
                    continue

                video_path    = os.path.join(video_dir, filename + ".mp4")
                skeleton_path = os.path.join(skeleton_dir, filename + ".npy")

                if not os.path.exists(video_path):
                    continue

                gloss = top_glosses[label_id] if label_id < len(top_glosses) else f"class_{label_id}"
                samples.append({
                    "video_path":    video_path,
                    "skeleton_path": skeleton_path,
                    "gloss_label":   label_id,
                    "gloss":         gloss,
                    "phon_labels":   [-1] * 16,  # AUTSL is TSL; no ASL-LEX coverage
                })

        out = {"num_classes": num_classes, "class_names": top_glosses, "samples": samples}
        out_path = os.path.join(output_dir, f"{sn}_annotations.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[AUTSL] {sn}: {len(samples)} samples -> {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build HiSLR annotation files")
    parser.add_argument("--dataset",      type=str, required=True, choices=["wlasl", "msasl", "autsl"])
    parser.add_argument("--video_dir",    type=str, required=True)
    parser.add_argument("--skeleton_dir", type=str, required=True)
    parser.add_argument("--split_file",   type=str, required=True, help="Path to split JSON or CSV directory")
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--num_classes",  type=int, default=2000)
    parser.add_argument("--asl_lex",      type=str, default="", help="Path to ASL-LEX JSON for phonological labels")
    args = parser.parse_args()

    asl_lex = load_asl_lex(args.asl_lex)

    builders = {
        "wlasl": build_wlasl,
        "msasl": build_msasl,
        "autsl": build_autsl,
    }
    builders[args.dataset](
        video_dir=args.video_dir,
        skeleton_dir=args.skeleton_dir,
        split_file=args.split_file,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        asl_lex=asl_lex,
    )
    print("Annotation build complete.")


if __name__ == "__main__":
    main()
