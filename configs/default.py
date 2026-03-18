# hislr/configs/default.py
# ─────────────────────────────────────────────────────────────────────────────
# HiSLR — Default Configuration
# All hyperparameters for the full model, training, and evaluation.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # Dataset
    dataset: str = "wlasl"                  # "wlasl" | "msasl" | "autsl"
    data_root: str = "./data/wlasl"
    annotation_file: str = "wlasl_annotations.json"
    asl_lex_file: str = "./data/asl_lex.json"  # phonological annotations
    num_classes: int = 2000                  # WLASL2000

    # Clip sampling
    num_frames: int = 16                    # T
    frame_size: int = 224                   # H, W
    sampling_strategy: str = "uniform"      # "uniform" | "random"

    # Skeleton
    num_joints: int = 75                    # 33 body + 21L + 21R hand
    joint_dim: int = 4                      # x, y, z, visibility
    normalize_skeleton: bool = True         # anchor-based normalization

    # Augmentation (train)
    random_flip: bool = True
    color_jitter: bool = True
    random_crop: bool = True
    joint_noise_sigma: float = 0.005
    joint_dropout_prob: float = 0.10
    speed_aug_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    skeleton_scale_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    skeleton_rotate_range: float = 15.0     # degrees

    # Data loader
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class ModelConfig:
    # ── RGB Encoder (Video Swin-V2) ──────────────────────────────────────────
    swin_variant: str = "swin_v2_b"         # "swin_v2_t" | "swin_v2_s" | "swin_v2_b"
    swin_pretrained: bool = True
    swin_pretrain_path: str = ""            # path to Kinetics-400 checkpoint
    swin_drop_path_rate: float = 0.2

    # Stage output channels [R1, R2, R3, R4]
    swin_channels: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])

    # ── Skeleton Encoder (MS-GCN) ────────────────────────────────────────────
    gcn_in_channels: int = 4               # joint_dim
    gcn_hidden_channels: List[int] = field(default_factory=lambda: [128, 256, 512])
    gcn_num_scales: int = 3                 # MS-G3D scales
    gcn_dropout: float = 0.25
    gcn_adaptive_graph: bool = True         # learnable intra-hand edge weights
    num_joints: int = 75

    # ── HiCMF Fusion ────────────────────────────────────────────────────────
    fusion_stages: int = 3                  # 1 | 2 | 3
    fusion_heads: List[int] = field(default_factory=lambda: [8, 8, 16])
    fusion_dropout: float = 0.1
    fusion_embed_dim: int = 256             # intermediate projection dim

    # Joint embedding dimension (after final fusion)
    joint_embed_dim: int = 2048

    # ── Classification Head ──────────────────────────────────────────────────
    cls_hidden_dim: int = 1024
    cls_dropout: float = 0.3
    label_smoothing: float = 0.1

    # ── Phonological Heads (PhAP) ────────────────────────────────────────────
    use_phap: bool = True
    phon_attributes: List[str] = field(default_factory=lambda: [
        "dominant_handshape",       # 87 classes
        "nondominant_handshape",    # 87 classes
        "major_location",           # 9 classes
        "minor_location",           # 32 classes
        "contact",                  # 4 classes
        "dominant_movement",        # 17 classes
        "path_movement",            # 6 classes
        "wrist_twist",              # 4 classes
        "palm_orientation",         # 6 classes
        "nmm_brows",                # 2 classes (binary)
        "nmm_eyes",                 # 2 classes
        "nmm_cheeks",               # 2 classes
        "nmm_mouth",                # 2 classes
        "nmm_tongue",               # 2 classes
        "nmm_head",                 # 2 classes
        "nmm_shoulders",            # 2 classes
    ])
    phon_num_classes: List[int] = field(default_factory=lambda: [
        87, 87, 9, 32, 4, 17, 6, 4, 6, 2, 2, 2, 2, 2, 2, 2
    ])

    # ── TCR (Temporal Contrastive Regularization) ────────────────────────────
    use_tcr: bool = True
    tcr_temperature: float = 0.07
    tcr_slow_factor: float = 0.5           # subsample rate for slow view
    tcr_fast_factor: float = 2.0           # oversample rate for fast view
    tcr_proj_dim: int = 256                # projection head output dim


@dataclass
class TrainingConfig:
    # Two-stage training
    stage1_epochs: int = 100               # PhAP pre-training (no TCR)
    stage2_epochs: int = 150               # Full training (all losses)
    total_epochs: int = 250

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0

    # Learning rate
    lr_new_layers: float = 1e-4
    lr_pretrained: float = 1e-5
    warmup_epochs: int = 10
    lr_scheduler: str = "cosine"           # "cosine" | "step"

    # Batch
    batch_size: int = 32                   # per GPU
    num_gpus: int = 4
    effective_batch_size: int = 128        # batch_size * num_gpus

    # Loss weights
    lambda_phon: float = 0.3
    lambda_tcr: float = 0.1

    # Mixed precision
    use_amp: bool = True
    use_grad_checkpoint: bool = True       # save memory for large models

    # Class imbalance
    sampling_strategy: str = "sqrt"        # "uniform" | "sqrt" | "inverse"

    # Logging & checkpointing
    log_interval: int = 50                 # steps
    val_interval: int = 1                  # epochs
    save_top_k: int = 3
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Reproducibility
    seed: int = 42
    num_seeds: int = 3                     # report mean ± 95% CI


@dataclass
class HiSLRConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment
    exp_name: str = "hislr_wlasl2000"
    output_dir: str = "./outputs"


def get_wlasl100_config() -> HiSLRConfig:
    cfg = HiSLRConfig()
    cfg.data.num_classes = 100
    cfg.data.dataset = "wlasl"
    cfg.exp_name = "hislr_wlasl100"
    return cfg


def get_wlasl2000_config() -> HiSLRConfig:
    cfg = HiSLRConfig()
    cfg.data.num_classes = 2000
    cfg.data.dataset = "wlasl"
    cfg.exp_name = "hislr_wlasl2000"
    return cfg


def get_msasl_config() -> HiSLRConfig:
    cfg = HiSLRConfig()
    cfg.data.num_classes = 1000
    cfg.data.dataset = "msasl"
    cfg.data.data_root = "./data/msasl"
    cfg.exp_name = "hislr_msasl1000"
    return cfg


def get_autsl_config() -> HiSLRConfig:
    cfg = HiSLRConfig()
    cfg.data.num_classes = 226
    cfg.data.dataset = "autsl"
    cfg.data.data_root = "./data/autsl"
    cfg.exp_name = "hislr_autsl"
    return cfg
