# kaggle_train.py
# ─────────────────────────────────────────────────────────────────────────────
# Chạy toàn bộ HiSLR pipeline trên Kaggle
# Cách dùng: chạy từng cell theo thứ tự trong notebook
# ─────────────────────────────────────────────────────────────────────────────

# ===========================================================================
# CELL 1 — Cài thư viện
# ===========================================================================
# %%
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("timm>=0.9.0")
pip("mediapipe")
pip("decord")
pip("opencv-python-headless")


# ===========================================================================
# CELL 2 — Kiểm tra GPU và đường dẫn
# ===========================================================================
# %%
import os, torch

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name    :", torch.cuda.get_device_name(0))
    print("GPU memory  :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

# ── Đường dẫn cố định — KHÔNG cần chỉnh sửa ─────────────────────────────────
DATA_ROOT    = "/kaggle/input/datasets/thtrnphc/wlasl100-new/WLASL_100"
WORK_DIR     = "/kaggle/working/HiSLR"
SKELETON_DIR = f"{WORK_DIR}/skeletons"
ANN_DIR      = f"{WORK_DIR}/annotations"
OUTPUT_DIR   = f"{WORK_DIR}/outputs"

# /kaggle/working/HiSLR chính là package hislr (git clone thẳng vào đây)
# nên sys.path phải trỏ vào THƯ MỤC CHA: /kaggle/working
CODE_DIR = "/kaggle/working"

for d in [SKELETON_DIR, ANN_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"DATA_ROOT : {DATA_ROOT}")
print(f"WORK_DIR  : {WORK_DIR}")

# Kiểm tra cấu trúc data
for split in ["train", "val", "test"]:
    split_path = f"{DATA_ROOT}/{split}"
    if os.path.exists(split_path):
        classes = [d for d in os.listdir(split_path) if os.path.isdir(f"{split_path}/{d}")]
        print(f"✅ {split}: {len(classes)} classes")
    else:
        print(f"⚠️  Không tìm thấy: {split_path}")


# ===========================================================================
# CELL 3 — Import HiSLR
# ===========================================================================
# %%
import sys, os

# Thêm /kaggle/working vào sys.path
# Python sẽ tìm thấy /kaggle/working/HiSLR → import được là "HiSLR"
# Nhưng tên package trong code là "hislr" (chữ thường)
# → Tạo symlink HiSLR → hislr để khớp tên
if not os.path.exists("/kaggle/working/hislr"):
    os.symlink("/kaggle/working/HiSLR", "/kaggle/working/hislr")
    print("✅ Tạo symlink hislr → HiSLR")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Test import
try:
    from hislr.configs.default import get_wlasl100_config
    print("✅ HiSLR import OK")
except ImportError as e:
    print(f"❌ Import lỗi: {e}")


# ===========================================================================
# CELL 4 — Extract skeleton (MediaPipe) — ~30–60 phút cho WLASL_100
# ===========================================================================
# %%
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

# Fix cho mediapipe >= 0.10.13
try:
    mp_holistic = mp.solutions.holistic
    print("mediapipe:", mp.__version__, "→ dùng mp.solutions")
except AttributeError:
    import mediapipe.python.solutions.holistic as mp_holistic
    print("mediapipe:", mp.__version__, "→ dùng mediapipe.python.solutions")

def extract_skeleton_from_video(video_path: str, output_path: str) -> bool:
    """Extract MediaPipe Holistic keypoints từ 1 video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    frames_joints = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = holistic.process(frame_rgb)

            joints = np.zeros((75, 4), dtype=np.float32)

            if results.pose_landmarks:
                for j, lm in enumerate(results.pose_landmarks.landmark):
                    joints[j] = [lm.x, lm.y, lm.z, lm.visibility]
            if results.right_hand_landmarks:
                for j, lm in enumerate(results.right_hand_landmarks.landmark):
                    joints[33 + j] = [lm.x, lm.y, lm.z, 1.0]
            if results.left_hand_landmarks:
                for j, lm in enumerate(results.left_hand_landmarks.landmark):
                    joints[54 + j] = [lm.x, lm.y, lm.z, 1.0]

            frames_joints.append(joints)

    cap.release()
    if not frames_joints:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.stack(frames_joints))  # (T, 75, 4)
    return True


# Thu thập tất cả video cần extract
all_videos = []
for split in ["train", "val", "test"]:
    split_dir = Path(DATA_ROOT) / split
    if not split_dir.exists():
        continue
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for video_file in sorted(class_dir.iterdir()):
            if video_file.suffix.lower() in {".mp4", ".avi", ".mov"}:
                rel = video_file.relative_to(DATA_ROOT)
                out = Path(SKELETON_DIR) / rel.with_suffix(".npy")
                if not out.exists():
                    all_videos.append((str(video_file), str(out)))

print(f"Cần extract: {len(all_videos)} videos")

failed = []
for video_path, out_path in tqdm(all_videos, desc="Extracting skeletons"):
    ok = extract_skeleton_from_video(video_path, out_path)
    if not ok:
        failed.append(video_path)

print(f"\n✅ Xong! Thất bại: {len(failed)}")


# ===========================================================================
# CELL 5 — Build annotation JSON
# ===========================================================================
# %%
import json
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

train_dir    = Path(DATA_ROOT) / "train"
class_names  = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
class_to_idx = {name: i for i, name in enumerate(class_names)}
num_classes  = len(class_names)
print(f"Số classes: {num_classes}")
print(f"Ví dụ: {class_names[:8]}")

for split in ["train", "val", "test"]:
    split_dir = Path(DATA_ROOT) / split
    if not split_dir.exists():
        continue

    samples = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in class_to_idx:
            continue

        label = class_to_idx[class_dir.name]
        for video_file in sorted(class_dir.iterdir()):
            if video_file.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            rel           = video_file.relative_to(DATA_ROOT)
            skeleton_path = str(Path(SKELETON_DIR) / rel.with_suffix(".npy"))

            samples.append({
                "video_path":    str(video_file),
                "skeleton_path": skeleton_path,
                "gloss_label":   label,
                "gloss":         class_dir.name,
                "phon_labels":   [-1] * 16,
            })

    out_path = f"{ANN_DIR}/{split}_annotations.json"
    with open(out_path, "w") as f:
        json.dump({
            "num_classes": num_classes,
            "class_names": class_names,
            "samples":     samples,
        }, f)
    print(f"  ✅ {split}: {len(samples)} samples → {out_path}")


# ===========================================================================
# CELL 6 — Cấu hình model
# ===========================================================================
# %%
from hislr.configs.default import HiSLRConfig

cfg = HiSLRConfig()

cfg.data.dataset         = "wlasl"
cfg.data.data_root       = ANN_DIR
cfg.data.annotation_file = "train_annotations.json"
cfg.data.num_classes     = num_classes
cfg.data.num_frames      = 16
cfg.data.frame_size      = 224
cfg.data.num_workers     = 2

cfg.model.swin_variant        = "swin_v2_t"
cfg.model.swin_pretrained     = True
cfg.model.gcn_hidden_channels = [64, 128, 256]
cfg.model.fusion_stages       = 3
cfg.model.joint_embed_dim     = 512
cfg.model.cls_hidden_dim      = 256
cfg.model.use_phap            = False
cfg.model.use_tcr             = True

cfg.training.batch_size     = 8
cfg.training.stage1_epochs  = 30
cfg.training.stage2_epochs  = 50
cfg.training.total_epochs   = 80
cfg.training.lr_new_layers  = 1e-4
cfg.training.lr_pretrained  = 1e-5
cfg.training.warmup_epochs  = 5
cfg.training.use_amp        = True
cfg.training.num_workers    = 2
cfg.training.checkpoint_dir = f"{OUTPUT_DIR}/checkpoints"
cfg.training.log_dir        = f"{OUTPUT_DIR}/logs"
cfg.exp_name                = "hislr_wlasl100_kaggle"

print("Cấu hình:")
print(f"  DATA_ROOT      : {DATA_ROOT}")
print(f"  ANN_DIR        : {ANN_DIR}")
print(f"  checkpoint_dir : {cfg.training.checkpoint_dir}")
print(f"  Model          : {cfg.model.swin_variant} | fusion={cfg.model.fusion_stages}")
print(f"  Classes        : {cfg.data.num_classes}")
print(f"  Epochs         : {cfg.training.total_epochs} (S1={cfg.training.stage1_epochs}, S2={cfg.training.stage2_epochs})")
print(f"  Batch size     : {cfg.training.batch_size}")
print(f"  AMP            : {cfg.training.use_amp}")


# ===========================================================================
# CELL 7 — Chạy training
# ===========================================================================
# %%
from hislr.training.trainer import HiSLRTrainer

trainer = HiSLRTrainer(cfg, rank=0, world_size=1)
trainer.train()


# ===========================================================================
# CELL 8 — Đánh giá trên test set
# ===========================================================================
# %%
import glob

checkpoints = sorted(
    glob.glob(f"{OUTPUT_DIR}/checkpoints/*.pth"),
    key=lambda x: float(x.split("acc")[-1].replace(".pth", "")),
    reverse=True,
)

if checkpoints:
    best_ckpt = checkpoints[0]
    print(f"Best checkpoint: {best_ckpt}")

    from hislr.inference import HiSLRInference

    engine = HiSLRInference(
        checkpoint_path = best_ckpt,
        cfg             = cfg,
        class_names     = class_names,
        device          = "cuda",
    )
    results = engine.evaluate_dataset(
        annotation_file = f"{ANN_DIR}/test_annotations.json",
        batch_size      = 16,
    )
    print(f"\nKết quả test set:")
    print(f"  P-I Top-1: {results['pi_top1']:.2f}%")
    print(f"  P-I Top-5: {results['pi_top5']:.2f}%")
else:
    print("⚠️  Chưa có checkpoint nào! Hãy chạy Cell 7 trước.")