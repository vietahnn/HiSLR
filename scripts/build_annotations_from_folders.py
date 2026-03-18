#!/usr/bin/env python3
# scripts/build_annotations_from_folders.py
# ─────────────────────────────────────────────────────────────────────────────
# Build HiSLR annotation JSON từ cấu trúc folder kiểu:
#
#   WLASL_100/
#   ├── train/
#   │   ├── accident/  ← tên gloss
#   │   │   ├── 00626.mp4
#   │   │   └── 00627.mp4
#   │   └── apple/
#   ├── val/
#   └── test/
#
# Usage:
#   python scripts/build_annotations_from_folders.py \
#       --data_root  /kaggle/input/WLASL_100 \
#       --output_dir ./data/wlasl \
#       --skeleton_dir ./data/wlasl/skeletons
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import argparse
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def build_from_folders(
    data_root: str,
    output_dir: str,
    skeleton_dir: str,
    asl_lex_path: str = "",
):
    data_root    = Path(data_root)
    output_dir   = Path(output_dir)
    skeleton_dir = Path(skeleton_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Bước 1: Thu thập tất cả class từ thư mục train ──────────────────────
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục train tại: {train_dir}")

    class_names = sorted([
        d.name for d in train_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes  = len(class_names)
    print(f"Tìm thấy {num_classes} classes: {class_names[:5]}...")

    # ── Bước 2: Load ASL-LEX nếu có ─────────────────────────────────────────
    asl_lex = {}
    if asl_lex_path and Path(asl_lex_path).exists():
        with open(asl_lex_path) as f:
            raw = json.load(f)
        PHON_ATTRS = [
            "dominant_handshape", "nondominant_handshape", "major_location",
            "minor_location", "contact", "dominant_movement", "path_movement",
            "wrist_twist", "palm_orientation", "nmm_brows", "nmm_eyes",
            "nmm_cheeks", "nmm_mouth", "nmm_tongue", "nmm_head", "nmm_shoulders",
        ]
        for gloss, attrs in raw.items():
            asl_lex[gloss.lower()] = [attrs.get(a, -1) for a in PHON_ATTRS]
        print(f"Loaded ASL-LEX: {len(asl_lex)} glosses")
    else:
        print("Không có ASL-LEX → phon_labels sẽ là [-1]*16 (vẫn train được, chỉ tắt PhAP)")

    # ── Bước 3: Duyệt từng split ─────────────────────────────────────────────
    splits = ["train", "val", "test"]
    stats  = {}

    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"  Bỏ qua '{split}' — không tìm thấy thư mục")
            continue

        samples = []
        missing_skeleton = 0

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            gloss = class_dir.name
            if gloss not in class_to_idx:
                # class xuất hiện trong val/test nhưng không có trong train → bỏ qua
                continue

            label       = class_to_idx[gloss]
            phon_labels = asl_lex.get(gloss.lower(), [-1] * 16)

            for video_file in sorted(class_dir.iterdir()):
                if video_file.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                video_path = str(video_file)

                # Skeleton path: mirror cấu trúc folder, đổi đuôi thành .npy
                rel            = video_file.relative_to(data_root)
                skeleton_path  = str(skeleton_dir / rel.with_suffix(".npy"))

                if not Path(skeleton_path).exists():
                    missing_skeleton += 1
                    # Vẫn thêm vào — dataset sẽ dùng zeros nếu không tìm thấy

                samples.append({
                    "video_path":    video_path,
                    "skeleton_path": skeleton_path,
                    "gloss_label":   label,
                    "gloss":         gloss,
                    "phon_labels":   phon_labels,
                })

        out_data = {
            "num_classes": num_classes,
            "class_names": class_names,
            "samples":     samples,
        }

        out_path = output_dir / f"{split}_annotations.json"
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)

        stats[split] = len(samples)
        print(f"  [{split}] {len(samples)} samples → {out_path}"
              + (f"  (thiếu {missing_skeleton} skeleton)" if missing_skeleton else ""))

    # ── Bước 4: In tóm tắt ───────────────────────────────────────────────────
    print("\n✅ Hoàn thành!")
    print(f"   Classes : {num_classes}")
    for split, count in stats.items():
        print(f"   {split:5s}   : {count} samples")
    print(f"   Output  : {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str, required=True,
                        help="Thư mục gốc chứa train/val/test (ví dụ: /kaggle/input/WLASL_100)")
    parser.add_argument("--output_dir",   type=str, required=True,
                        help="Thư mục lưu file annotation JSON")
    parser.add_argument("--skeleton_dir", type=str, required=True,
                        help="Thư mục lưu file skeleton .npy")
    parser.add_argument("--asl_lex",      type=str, default="",
                        help="(Tùy chọn) Path đến file ASL-LEX JSON")
    args = parser.parse_args()

    build_from_folders(
        data_root    = args.data_root,
        output_dir   = args.output_dir,
        skeleton_dir = args.skeleton_dir,
        asl_lex_path = args.asl_lex,
    )


if __name__ == "__main__":
    main()
