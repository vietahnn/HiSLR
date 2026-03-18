#!/usr/bin/env python3
# train.py
# ─────────────────────────────────────────────────────────────────────────────
# HiSLR — Main Training Entry Point
#
# Usage:
#   # Single GPU
#   python train.py --dataset wlasl2000 --data_root ./data/wlasl
#
#   # Multi-GPU (4 GPUs)
#   torchrun --nproc_per_node=4 train.py --dataset wlasl2000 --data_root ./data/wlasl
#
#   # Resume from checkpoint
#   python train.py --dataset wlasl2000 --resume ./checkpoints/epoch_0050_acc0.6512.pth
#
#   # Ablation variants
#   python train.py --dataset wlasl2000 --fusion_stages 0 --exp_name hislr_lf
#   python train.py --dataset wlasl2000 --no_phap --exp_name hislr_nophap
#   python train.py --dataset wlasl2000 --no_tcr  --exp_name hislr_notcr
#   python train.py --dataset wlasl2000 --rgb_only --exp_name hislr_rgb
#   python train.py --dataset wlasl2000 --pose_only --exp_name hislr_pose
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist

from hislr.configs.default import (
    HiSLRConfig,
    get_wlasl100_config,
    get_wlasl2000_config,
    get_msasl_config,
    get_autsl_config,
)
from hislr.training.trainer import HiSLRTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="HiSLR Training")

    # Dataset
    parser.add_argument("--dataset", type=str, default="wlasl2000",
                        choices=["wlasl100", "wlasl300", "wlasl1000", "wlasl2000",
                                 "msasl100", "msasl500", "msasl1000", "autsl"],
                        help="Dataset and vocabulary size")
    parser.add_argument("--data_root", type=str, default="./data/wlasl")
    parser.add_argument("--asl_lex_file", type=str, default="./data/asl_lex.json")

    # Architecture ablations
    parser.add_argument("--fusion_stages", type=int, default=3,
                        help="Number of HiCMF fusion stages (0=late only, 1-3)")
    parser.add_argument("--swin_variant", type=str, default="swin_v2_b",
                        choices=["swin_v2_t", "swin_v2_s", "swin_v2_b", "swin_v2_l"])
    parser.add_argument("--no_phap",   action="store_true", help="Disable PhAP (no phonological heads)")
    parser.add_argument("--no_tcr",    action="store_true", help="Disable TCR loss")
    parser.add_argument("--rgb_only",  action="store_true", help="Use RGB stream only (no skeleton)")
    parser.add_argument("--pose_only", action="store_true", help="Use skeleton only (no RGB)")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage2_epochs", type=int, default=150)
    parser.add_argument("--lr_new",   type=float, default=1e-4)
    parser.add_argument("--lr_pretrained", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    # Logging
    parser.add_argument("--exp_name", type=str, default="hislr_wlasl2000")
    parser.add_argument("--output_dir", type=str, default="./outputs")

    # DDP
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_config(args) -> HiSLRConfig:
    """Build HiSLRConfig from command-line arguments."""
    dataset_map = {
        "wlasl100": get_wlasl100_config,
        "wlasl2000": get_wlasl2000_config,
        "msasl1000": get_msasl_config,
        "autsl": get_autsl_config,
    }

    # Use closest preset
    if args.dataset in dataset_map:
        cfg = dataset_map[args.dataset]()
    elif args.dataset.startswith("wlasl"):
        cfg = get_wlasl2000_config()
        n = int(args.dataset.replace("wlasl", ""))
        cfg.data.num_classes = n
        cfg.exp_name = f"hislr_{args.dataset}"
    elif args.dataset.startswith("msasl"):
        cfg = get_msasl_config()
        n = int(args.dataset.replace("msasl", ""))
        cfg.data.num_classes = n
        cfg.exp_name = f"hislr_{args.dataset}"
    else:
        cfg = HiSLRConfig()

    # Override from args
    cfg.data.data_root = args.data_root
    cfg.data.asl_lex_file = args.asl_lex_file
    cfg.model.swin_variant = args.swin_variant
    cfg.model.fusion_stages = args.fusion_stages if not (args.rgb_only or args.pose_only) else 0
    cfg.model.use_phap = not args.no_phap
    cfg.model.use_tcr  = not args.no_tcr and not args.pose_only
    cfg.training.batch_size = args.batch_size
    cfg.training.stage1_epochs = args.stage1_epochs
    cfg.training.stage2_epochs = args.stage2_epochs
    cfg.training.total_epochs = args.stage1_epochs + args.stage2_epochs
    cfg.training.lr_new_layers = args.lr_new
    cfg.training.lr_pretrained = args.lr_pretrained
    cfg.training.seed = args.seed
    cfg.exp_name = args.exp_name
    cfg.output_dir = args.output_dir
    cfg.training.checkpoint_dir = os.path.join(args.output_dir, args.exp_name, "checkpoints")
    cfg.training.log_dir = os.path.join(args.output_dir, args.exp_name, "logs")

    # RGB-only / Pose-only ablation flags are handled in trainer via model config
    # (For simplicity, the trainer detects these through fusion_stages=0 + encoder flags)
    if args.rgb_only:
        cfg.model.use_phap = False
        cfg.model.use_tcr  = False
    if args.pose_only:
        cfg.model.use_phap = False
        cfg.model.use_tcr  = False

    return cfg


def main():
    args = parse_args()

    # ── Distributed init ─────────────────────────────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", args.local_rank))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

    set_seed(args.seed + rank)

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = build_config(args)

    if rank == 0:
        print("=" * 70)
        print(f"  HiSLR Training — {cfg.exp_name}")
        print(f"  Dataset : {args.dataset} ({cfg.data.num_classes} classes)")
        print(f"  Stages  : HiCMF={cfg.model.fusion_stages}  PhAP={cfg.model.use_phap}  TCR={cfg.model.use_tcr}")
        print(f"  GPUs    : {world_size}  Batch/GPU: {cfg.training.batch_size}")
        print(f"  Epochs  : {cfg.training.total_epochs} (S1={cfg.training.stage1_epochs}, S2={cfg.training.stage2_epochs})")
        print("=" * 70)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = HiSLRTrainer(cfg, rank=rank, world_size=world_size)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        raw_model.load_state_dict(ckpt["model_state"], strict=False)
        if rank == 0:
            print(f"Resumed from {args.resume}")

    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
