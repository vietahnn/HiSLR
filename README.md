# HiSLR — Hierarchical Sign Language Recognition

Official source code for the proposed model:

> **HiSLR: A Hierarchical Multimodal Fusion Network with Phonological-Aware
> Pre-Training for Isolated Sign Language Recognition**

Projected WLASL2000 P-I Top-1: **73–76%** (vs. SOTA 67.07%)

---

## Architecture Overview

```
RGB Clip (B, T, 3, H, W)                Skeleton (B, T, J, 4)
        │                                       │
  ┌─────▼──────┐                        ┌──────▼──────┐
  │  Swin-V2-B │                        │   MS-GCN    │
  │  Encoder   │                        │  Encoder    │
  └─────┬──────┘                        └──────┬──────┘
        │ R1, R2, R3, R4                       │ S1, S2, S3
        │                                      │
        └──────────┐     ┌────────────────────┘
                   │     │
             ┌─────▼─────▼─────┐
             │    HiCMF         │  ◄── 3-stage bidirectional
             │  Fusion Backbone │       cross-modal attention
             └────────┬────────┘
                      │ z_joint (2048)
          ┌───────────┼───────────┐
          │           │           │
    ┌─────▼─────┐ ┌───▼───┐ ┌────▼────┐
    │ Gloss     │ │ PhAP  │ │  TCR   │
    │ Cls Head  │ │ Heads │ │ Proj   │
    └───────────┘ └───────┘ └────────┘
```

**Three core contributions:**
1. **HiCMF** — Hierarchical Cross-Modal Fusion: 3-stage bidirectional cross-attention between RGB and skeleton streams
2. **PhAP** — Phonological-Aware Pre-Training: 16 auxiliary phonological attribute heads (handshape, location, movement, orientation, NMMs)
3. **TCR** — Temporal Contrastive Regularization: InfoNCE loss between slow/fast temporal augmented views

---

## Project Structure

```
hislr/
├── configs/
│   └── default.py          # All hyperparameters (DataConfig, ModelConfig, TrainingConfig)
├── models/
│   ├── hislr.py            # Full HiSLR model (main entry point)
│   ├── hicmf.py            # Hierarchical Cross-Modal Fusion backbone
│   ├── msgcn.py            # Multi-Scale Spatio-Temporal GCN encoder
│   └── swin_wrapper.py     # Video Swin-V2 encoder wrapper
├── data/
│   └── dataset.py          # ISLRDataset, augmentations, weighted sampler
├── training/
│   ├── losses.py           # LabelSmoothCE, PhonologicalLoss, TCRLoss, HiSLRLoss
│   └── trainer.py          # HiSLRTrainer — full two-stage training engine
├── utils/
│   └── metrics.py          # AverageMeter, compute_accuracy, confusion matrix
├── inference.py            # HiSLRInference class for deployment
├── train.py                # Training entry point
├── evaluate.py             # Evaluation / demo entry point
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For Video Swin-V2:
```bash
pip install timm>=0.9.0
```

### 2. Prepare data

**WLASL:**
1. Download WLASL from https://github.com/dxli94/WLASL
2. Extract skeleton keypoints with MediaPipe Holistic:

```bash
python scripts/extract_skeletons.py \
    --video_dir ./data/wlasl/videos \
    --output_dir ./data/wlasl/skeletons
```

3. Build annotation JSON:
```bash
python scripts/build_annotations.py \
    --dataset wlasl \
    --video_dir ./data/wlasl/videos \
    --skeleton_dir ./data/wlasl/skeletons \
    --split_file ./data/wlasl/WLASL_v0.3.json \
    --output_dir ./data/wlasl
```

**Phonological annotations (PhAP):**
Download ASL-LEX 2.0 from https://asl-lex.org/ and place at `./data/asl_lex.json`.

---

## Training

### Single GPU
```bash
python -m hislr.train \
    --dataset wlasl2000 \
    --data_root ./data/wlasl \
    --exp_name hislr_wlasl2000
```

### Multi-GPU (4× A100)
```bash
torchrun --nproc_per_node=4 -m hislr.train \
    --dataset wlasl2000 \
    --data_root ./data/wlasl \
    --batch_size 32 \
    --exp_name hislr_wlasl2000
```

### Ablation variants
```bash
# Late fusion only (HiSLR-LF)
python -m hislr.train --dataset wlasl2000 --fusion_stages 0 --exp_name hislr_lf

# No PhAP (HiSLR-NoPhAP)
python -m hislr.train --dataset wlasl2000 --no_phap --exp_name hislr_nophap

# No TCR (HiSLR-NoTCR)
python -m hislr.train --dataset wlasl2000 --no_tcr --exp_name hislr_notcr

# RGB-only baseline
python -m hislr.train --dataset wlasl2000 --rgb_only --exp_name hislr_rgb

# Pose-only baseline
python -m hislr.train --dataset wlasl2000 --pose_only --exp_name hislr_pose
```

---

## Evaluation

```bash
# Full test-set evaluation
python -m hislr.evaluate \
    --checkpoint ./outputs/hislr_wlasl2000/checkpoints/best.pth \
    --dataset wlasl2000 \
    --data_root ./data/wlasl \
    --split test

# Single video demo
python -m hislr.evaluate \
    --checkpoint ./outputs/hislr_wlasl2000/checkpoints/best.pth \
    --demo \
    --video_path ./sample_sign.mp4 \
    --skeleton_path ./sample_sign.npy \
    --top_k 5
```

---

## Expected Results

| Dataset       | P-I Top-1 | P-I Top-5 |
|---------------|-----------|-----------|
| WLASL100      | ~95%      | ~99%      |
| WLASL300      | ~88%      | ~97%      |
| WLASL1000     | ~80%      | ~94%      |
| WLASL2000     | 73–76%    | ~90%      |
| MS-ASL1000    | 87–90%    | ~97%      |
| AUTSL         | ~95%      | ~99%      |

---

## Annotation File Format

```json
{
    "num_classes": 2000,
    "class_names": ["hello", "world", ...],
    "samples": [
        {
            "video_path": "data/wlasl/videos/00001.mp4",
            "skeleton_path": "data/wlasl/skeletons/00001.npy",
            "gloss_label": 0,
            "phon_labels": [3, 12, -1, 7, 2, 5, 1, 0, 3, 0, 1, 0, 1, 0, 0, 1]
        }
    ]
}
```

Skeleton `.npy` files: shape `(T, 75, 4)` — float32 array of `(x, y, z, visibility)` for 75 MediaPipe Holistic joints. Raw (unnormalized) coordinates; normalization is applied in the dataset pipeline.

`phon_labels`: 16 integers matching the 16 PhAP attributes in `ModelConfig.phon_attributes`. Use `-1` for signs not covered by ASL-LEX annotations.

---

## Citation

```bibtex
@article{hislr2026,
  title   = {HiSLR: A Hierarchical Multimodal Fusion Network with
             Phonological-Aware Pre-Training for Isolated Sign Language Recognition},
  year    = {2026},
}
```
