# hislr/data/dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# HiSLR Dataset & Augmentation Pipeline
#
# Handles:
#   - WLASL / MS-ASL / AUTSL video loading
#   - Skeleton keypoint loading (MediaPipe pre-extracted JSON)
#   - Anchor-based skeleton normalization
#   - All RGB and skeleton augmentations described in the paper
#   - TCR temporal augmentation (slow/fast view generation)
#   - Phonological annotation loading from ASL-LEX
#   - Square-root class balanced sampling
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import Optional, Tuple, List
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ── Skeleton normalization ─────────────────────────────────────────────────────

# Shoulder joint indices in the 75-joint skeleton
LEFT_SHOULDER_IDX  = 11
RIGHT_SHOULDER_IDX = 12


def normalize_skeleton(joints: np.ndarray) -> np.ndarray:
    """
    Anchor-based normalization. Translates all joints relative to the
    shoulder midpoint and scales by shoulder width.

    Args:
        joints: (T, J, 4) — (x, y, z, visibility)
    Returns:
        joints_norm: (T, J, 4) — normalized, visibility unchanged
    """
    left  = joints[:, LEFT_SHOULDER_IDX,  :3]   # (T, 3)
    right = joints[:, RIGHT_SHOULDER_IDX, :3]   # (T, 3)

    anchor = (left + right) / 2.0               # shoulder midpoint (T, 3)
    width  = np.linalg.norm(left - right, axis=-1, keepdims=True).clip(1e-6)  # (T, 1)

    xyz = joints[:, :, :3]                      # (T, J, 3)
    xyz_norm = (xyz - anchor[:, None, :]) / width[:, None, :]

    result = joints.copy()
    result[:, :, :3] = xyz_norm
    return result


# ── Temporal augmentation helpers ─────────────────────────────────────────────

def temporal_subsample(frames: np.ndarray, joints: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Temporally subsample (slow) or oversample (fast) a clip.
    Returns clip of the same length T by resampling indices.

    Args:
        frames: (T, H, W, 3)
        joints: (T, J, 4)
        factor: <1 = slow (use fewer original frames), >1 = fast (denser sampling)
    Returns:
        frames_aug: (T, H, W, 3)
        joints_aug: (T, J, 4)
    """
    T = frames.shape[0]
    total = max(1, int(T / factor))  # how many frames to actually use
    indices = np.linspace(0, T - 1, total).astype(int)
    # Resample back to T
    resample_idx = np.round(np.linspace(0, len(indices) - 1, T)).astype(int)
    final_idx = indices[resample_idx]

    return frames[final_idx], joints[final_idx]


# ── RGB augmentation ───────────────────────────────────────────────────────────

class VideoAugmentor:
    """
    Apply per-clip RGB augmentations consistent across all frames.
    """

    def __init__(
        self,
        is_train: bool = True,
        frame_size: int = 224,
        color_jitter: bool = True,
        random_flip: bool = True,
    ):
        self.is_train = is_train
        self.frame_size = frame_size

        if is_train:
            self.color_jitter = T.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
            ) if color_jitter else None
            self.random_flip = random_flip
        else:
            self.color_jitter = None
            self.random_flip = False

    def __call__(self, frames: torch.Tensor, flip: bool = False) -> torch.Tensor:
        """
        Args:
            frames: (T, 3, H, W)
            flip:   whether to flip horizontally
        Returns:
            frames: (T, 3, H, W) augmented
        """
        if self.is_train:
            # Consistent color jitter (same params for all frames)
            if self.color_jitter is not None:
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                    T.ColorJitter.get_params(
                        self.color_jitter.brightness,
                        self.color_jitter.contrast,
                        self.color_jitter.saturation,
                        self.color_jitter.hue,
                    )
                for fn_id in fn_idx:
                    if fn_id == 0:
                        frames = TF.adjust_brightness(frames, brightness_factor)
                    elif fn_id == 1:
                        frames = TF.adjust_contrast(frames, contrast_factor)
                    elif fn_id == 2:
                        frames = TF.adjust_saturation(frames, saturation_factor)
                    elif fn_id == 3:
                        frames = TF.adjust_hue(frames, hue_factor)

            # Horizontal flip
            if flip:
                frames = TF.hflip(frames)

        return frames


# ── Skeleton augmentation ──────────────────────────────────────────────────────

def augment_skeleton(
    joints: np.ndarray,
    noise_sigma: float = 0.005,
    dropout_prob: float = 0.10,
    scale_range: tuple = (0.9, 1.1),
    rotate_range: float = 15.0,
    flip: bool = False,
    num_joints: int = 75,
) -> np.ndarray:
    """
    Apply all skeleton augmentations to a (T, J, 4) skeleton array.

    Args:
        joints:       (T, J, 4)
        noise_sigma:  std of Gaussian noise added to XYZ
        dropout_prob: probability of zeroing each joint per frame
        scale_range:  global scale multiplier range
        rotate_range: frontal-plane rotation range in degrees
        flip:         mirror left-right (requires joint index remapping)
        num_joints:   total number of joints

    Returns:
        joints_aug: (T, J, 4)
    """
    joints = joints.copy()

    # Joint noise
    noise = np.random.normal(0, noise_sigma, joints[:, :, :3].shape)
    joints[:, :, :3] += noise

    # Joint dropout
    drop_mask = np.random.rand(*joints.shape[:2]) < dropout_prob  # (T, J)
    joints[drop_mask, :3] = 0.0
    joints[drop_mask,  3] = 0.0  # visibility = 0 for dropped joints

    # Global scale
    scale = np.random.uniform(*scale_range)
    joints[:, :, :3] *= scale

    # Frontal-plane rotation (around Z axis)
    angle_deg = np.random.uniform(-rotate_range, rotate_range)
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    x = joints[:, :, 0].copy()
    y = joints[:, :, 1].copy()
    joints[:, :, 0] = cos_a * x - sin_a * y
    joints[:, :, 1] = sin_a * x + cos_a * y

    # Horizontal flip: swap left/right hand and body joints
    if flip:
        # Body: swap left/right pairs (simplified — full mapping below)
        LR_BODY_PAIRS = [
            (1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
            (17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)
        ]
        for l, r in LR_BODY_PAIRS:
            joints[:, [l, r]] = joints[:, [r, l]]
        # Swap entire left/right hand blocks
        RH_BASE, LH_BASE = 33, 54
        for j in range(21):
            joints[:, [RH_BASE+j, LH_BASE+j]] = joints[:, [LH_BASE+j, RH_BASE+j]]

        # Mirror X coordinate
        joints[:, :, 0] *= -1

    return joints


# ── Main Dataset class ─────────────────────────────────────────────────────────

class ISLRDataset(Dataset):
    """
    Isolated Sign Language Recognition Dataset.

    Expects a JSON annotation file with the following structure:
    {
        "samples": [
            {
                "video_path": "path/to/video.mp4",
                "skeleton_path": "path/to/skeleton.npy",  // (T, J, 4) numpy array
                "gloss_label": 42,                         // integer class index
                "phon_labels": [3, 12, -1, 7, ...]        // 16 ints, -1 = missing
            },
            ...
        ],
        "class_names": ["hello", "world", ...],
        "num_classes": 2000
    }
    """

    def __init__(
        self,
        annotation_file: str,
        num_frames: int = 16,
        frame_size: int = 224,
        num_joints: int = 75,
        is_train: bool = True,
        normalize_skeleton: bool = True,
        # Augmentation flags
        random_flip: bool = True,
        color_jitter: bool = True,
        joint_noise_sigma: float = 0.005,
        joint_dropout_prob: float = 0.10,
        speed_aug_range: tuple = (0.8, 1.2),
        skeleton_scale_range: tuple = (0.9, 1.1),
        skeleton_rotate_range: float = 15.0,
        # TCR
        tcr_slow_factor: float = 0.5,
        tcr_fast_factor: float = 2.0,
        generate_tcr_views: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_joints = num_joints
        self.is_train = is_train
        self.do_normalize = normalize_skeleton
        self.generate_tcr_views = generate_tcr_views
        self.tcr_slow_factor = tcr_slow_factor
        self.tcr_fast_factor = tcr_fast_factor

        # Augmentation params
        self.aug_params = dict(
            random_flip=random_flip and is_train,
            color_jitter=color_jitter and is_train,
            noise_sigma=joint_noise_sigma,
            dropout_prob=joint_dropout_prob,
            speed_aug_range=speed_aug_range,
            scale_range=skeleton_scale_range,
            rotate_range=skeleton_rotate_range,
        )

        with open(annotation_file, "r") as f:
            ann = json.load(f)

        self.samples = ann["samples"]
        self.class_names = ann.get("class_names", [])
        self.num_classes = ann.get("num_classes", len(self.class_names))

        # ImageNet mean/std for normalization
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.rgb_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.video_aug = VideoAugmentor(
            is_train=is_train,
            frame_size=frame_size,
            color_jitter=color_jitter and is_train,
            random_flip=random_flip and is_train,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, video_path: str) -> np.ndarray:
        """
        Load T frames from a video file, uniformly sampled.
        Returns: (T, H, W, 3) uint8 numpy array.

        NOTE: Requires either `decord` or `torchvision.io.read_video`.
        Falls back to a zeros array for environments without video IO.
        """
        try:
            import decord
            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(video_path)
            total = len(vr)
            indices = np.linspace(0, total - 1, self.num_frames).astype(int)
            frames = vr.get_batch(indices).numpy()  # (T, H, W, 3)
        except ImportError:
            try:
                from torchvision.io import read_video
                video, _, _ = read_video(video_path, pts_unit="sec")
                total = video.shape[0]
                indices = np.linspace(0, total - 1, self.num_frames).astype(int)
                frames = video[indices].numpy()  # (T, H, W, 3)
            except Exception:
                # Fallback for testing without video files
                frames = np.zeros((self.num_frames, self.frame_size, self.frame_size, 3), dtype=np.uint8)

        # Resize to frame_size
        if frames.shape[1] != self.frame_size or frames.shape[2] != self.frame_size:
            import cv2
            resized = []
            for f in frames:
                resized.append(cv2.resize(f, (self.frame_size, self.frame_size)))
            frames = np.stack(resized)

        return frames

    def _load_skeleton(self, skeleton_path: str) -> np.ndarray:
        """
        Load pre-extracted skeleton keypoints.
        Expected format: (T_orig, J, 4) numpy array, saved as .npy.
        Returns (num_frames, J, 4) after temporal resampling.
        """
        try:
            joints = np.load(skeleton_path).astype(np.float32)  # (T_orig, J, 4)
        except Exception:
            joints = np.zeros((self.num_frames, self.num_joints, 4), dtype=np.float32)
            return joints

        T_orig = joints.shape[0]
        if T_orig != self.num_frames:
            indices = np.round(np.linspace(0, T_orig - 1, self.num_frames)).astype(int)
            joints = joints[indices]

        return joints

    def _frames_to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """
        Convert (T, H, W, 3) uint8 numpy to (T, 3, H, W) float32 tensor,
        normalized with ImageNet mean/std.
        """
        t = torch.from_numpy(frames).float() / 255.0  # (T, H, W, 3)
        t = t.permute(0, 3, 1, 2)                     # (T, 3, H, W)
        t = (t - self.rgb_mean) / self.rgb_std
        return t

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── Load raw data ─────────────────────────────────────────────────────
        frames = self._load_frames(sample["video_path"])    # (T, H, W, 3)
        joints = self._load_skeleton(sample["skeleton_path"])  # (T, J, 4)

        gloss_label = sample["gloss_label"]
        phon_labels = sample.get("phon_labels", [-1] * 16)

        # ── Speed augmentation (train only) ───────────────────────────────────
        if self.is_train:
            speed_factor = random.uniform(*self.aug_params["speed_aug_range"])
            if abs(speed_factor - 1.0) > 0.05:
                frames, joints = temporal_subsample(frames, joints, speed_factor)

        # ── Skeleton normalization ─────────────────────────────────────────────
        if self.do_normalize:
            joints = normalize_skeleton(joints)

        # ── Determine flip ────────────────────────────────────────────────────
        do_flip = self.is_train and self.aug_params["random_flip"] and random.random() < 0.5

        # ── Skeleton augmentation ─────────────────────────────────────────────
        if self.is_train:
            joints = augment_skeleton(
                joints,
                noise_sigma=self.aug_params["noise_sigma"],
                dropout_prob=self.aug_params["dropout_prob"],
                scale_range=self.aug_params["scale_range"],
                rotate_range=self.aug_params["rotate_range"],
                flip=do_flip,
            )

        # ── RGB augmentation & tensor conversion ──────────────────────────────
        rgb_tensor = self._frames_to_tensor(frames)        # (T, 3, H, W)
        rgb_tensor = self.video_aug(rgb_tensor, flip=do_flip)

        joints_tensor = torch.from_numpy(joints).float()  # (T, J, 4)

        out = {
            "rgb":         rgb_tensor,                                         # (T, 3, H, W)
            "skeleton":    joints_tensor,                                      # (T, J, 4)
            "gloss_label": torch.tensor(gloss_label, dtype=torch.long),
            "phon_labels": torch.tensor(phon_labels, dtype=torch.long),        # (16,)
            "index":       idx,
        }

        # ── TCR slow/fast views ───────────────────────────────────────────────
        if self.generate_tcr_views and self.is_train:
            frames_raw = self._load_frames(sample["video_path"])
            joints_raw = self._load_skeleton(sample["skeleton_path"])
            if self.do_normalize:
                joints_raw = normalize_skeleton(joints_raw)

            # Slow view
            frames_slow, joints_slow = temporal_subsample(
                frames_raw, joints_raw, self.tcr_slow_factor
            )
            joints_slow = augment_skeleton(joints_slow, noise_sigma=self.aug_params["noise_sigma"])
            out["rgb_slow"]      = self.video_aug(self._frames_to_tensor(frames_slow))
            out["skeleton_slow"] = torch.from_numpy(joints_slow).float()

            # Fast view
            frames_fast, joints_fast = temporal_subsample(
                frames_raw, joints_raw, self.tcr_fast_factor
            )
            joints_fast = augment_skeleton(joints_fast, noise_sigma=self.aug_params["noise_sigma"])
            out["rgb_fast"]      = self.video_aug(self._frames_to_tensor(frames_fast))
            out["skeleton_fast"] = torch.from_numpy(joints_fast).float()

        return out


def build_weighted_sampler(dataset: ISLRDataset, strategy: str = "sqrt") -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler for class-imbalanced training.

    Strategies:
        "uniform":  standard random sampling (no reweighting)
        "sqrt":     weight proportional to sqrt(freq)  [recommended]
        "inverse":  weight proportional to 1/freq
    """
    labels = [s["gloss_label"] for s in dataset.samples]
    num_classes = dataset.num_classes

    # Compute class frequencies
    freq = np.zeros(num_classes, dtype=np.float64)
    for l in labels:
        freq[l] += 1

    if strategy == "inverse":
        class_weight = 1.0 / np.maximum(freq, 1)
    elif strategy == "sqrt":
        class_weight = 1.0 / np.maximum(np.sqrt(freq), 1e-6)
    else:  # uniform
        class_weight = np.ones(num_classes)

    sample_weights = torch.tensor([class_weight[l] for l in labels], dtype=torch.float64)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
