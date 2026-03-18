# hislr/models/msgcn.py
# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Spatio-Temporal Graph Convolutional Network (MS-GCN)
# Skeleton encoder for HiSLR.
#
# Architecture:
#   - Hand-aware adaptive graph topology (anatomical + learnable intra-hand edges)
#   - 3 MS-GCN stages producing features at T, T/2, T/4 temporal resolutions
#   - Each stage: Graph Conv → Temporal Conv → BN → ReLU
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Joint topology definitions ────────────────────────────────────────────────

# MediaPipe Holistic joint indices (75 total)
# 0–32  : body (33 joints)
# 33–53 : left hand (21 joints)
# 54–74 : right hand (21 joints)

BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),     # face chain
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),                              # mouth
    (11, 12), (11, 13), (13, 15),        # left arm
    (12, 14), (14, 16),                  # right arm
    (15, 17), (15, 19), (15, 21),        # left hand attach
    (16, 18), (16, 20), (16, 22),        # right hand attach
    (11, 23), (12, 24),                  # torso top
    (23, 24), (23, 25), (24, 26),        # torso bottom
    (25, 27), (26, 28),                  # legs
    (27, 29), (27, 31), (29, 31),        # left foot
    (28, 30), (28, 32), (30, 32),        # right foot
]

# Hand finger chains (relative to hand base index)
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],    # thumb
    [0, 5, 6, 7, 8],    # index
    [0, 9, 10, 11, 12], # middle
    [0, 13, 14, 15, 16],# ring
    [0, 17, 18, 19, 20],# pinky
]

CROSS_BODY_EDGES = [
    (15, 54),  # left wrist -> left hand base
    (16, 33),  # right wrist -> right hand base (adjusted indices)
    (33, 54),  # right hand wrist to left hand wrist (coordination)
    (11, 33),  # left shoulder to right hand
    (12, 54),  # right shoulder to left hand
    (0, 33),   # nose to right hand
    (0, 54),   # nose to left hand
]


def build_adjacency_matrix(num_joints: int = 75) -> np.ndarray:
    """
    Build the initial adjacency matrix for the 75-joint graph.
    Returns a (num_joints, num_joints) binary adjacency matrix.
    """
    A = np.zeros((num_joints, num_joints), dtype=np.float32)

    def add_edges(edges, offset_i=0, offset_j=0):
        for i, j in edges:
            A[i + offset_i, j + offset_j] = 1
            A[j + offset_j, i + offset_i] = 1

    # Body edges
    add_edges(BODY_EDGES)

    # Right hand edges (base index = 33)
    rh_base = 33
    for chain in FINGER_CHAINS:
        for k in range(len(chain) - 1):
            a, b = rh_base + chain[k], rh_base + chain[k + 1]
            A[a, b] = A[b, a] = 1

    # Left hand edges (base index = 54)
    lh_base = 54
    for chain in FINGER_CHAINS:
        for k in range(len(chain) - 1):
            a, b = lh_base + chain[k], lh_base + chain[k + 1]
            A[a, b] = A[b, a] = 1

    # Cross-body edges
    add_edges(CROSS_BODY_EDGES)

    # Self-connections
    np.fill_diagonal(A, 1)

    # Normalize: D^{-1/2} A D^{-1/2}
    D = np.diag(A.sum(axis=1) ** -0.5)
    A = D @ A @ D

    return A


class SpatialGraphConv(nn.Module):
    """
    Single-scale spatial graph convolution.
    Applies A * X * W where A is the (possibly adaptive) adjacency matrix.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = 75,
        adaptive: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.adaptive = adaptive

        # Base adjacency (normalized, fixed)
        A = build_adjacency_matrix(num_joints)
        self.register_buffer("A_base", torch.from_numpy(A))

        # Learnable adaptive adjacency (added to base)
        if adaptive:
            self.A_learn = nn.Parameter(torch.zeros(num_joints, num_joints))
            nn.init.xavier_uniform_(self.A_learn)

        # Weight matrix
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, C_in)
        Returns:
            out: (B, T, J, C_out)
        """
        B, T, J, C = x.shape

        A = self.A_base
        if self.adaptive:
            # Soft-max adaptive adjacency
            A = A + torch.softmax(self.A_learn, dim=-1)

        # x: (B*T, J, C) -> graph conv -> (B*T, J, C_out)
        x_flat = x.reshape(B * T, J, C)

        # AX: (J, J) x (B*T, J, C) -> (B*T, J, C)
        Ax = torch.einsum("jk,bkc->bjc", A, x_flat)  # (B*T, J, C)

        # Linear transform
        out = self.W(Ax)  # (B*T, J, C_out)

        # BN over channels
        out = out.reshape(B * T * J, -1)
        out = self.bn(out)
        out = out.reshape(B, T, J, -1)

        return F.relu(out, inplace=True)


class MultiScaleGraphConv(nn.Module):
    """
    Multi-Scale Graph Convolution (MS-GCN stage).
    Combines features at K different neighborhood scales by computing
    A^1 X W_1 + A^2 X W_2 + ... + A^K X W_K.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = 75,
        num_scales: int = 3,
        adaptive: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_scales = num_scales

        # One graph conv per scale
        self.convs = nn.ModuleList([
            SpatialGraphConv(in_channels, out_channels // num_scales, num_joints, adaptive)
            for _ in range(num_scales)
        ])

        # Residual connection
        self.residual = (
            nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_powers: list) -> torch.Tensor:
        """
        Args:
            x       : (B, T, J, C_in)
            A_powers: list of K adjacency tensors A^k each (J, J)
        Returns:
            out: (B, T, J, C_out)
        """
        B, T, J, C = x.shape
        parts = []

        x_flat = x.reshape(B * T, J, C)

        for k, conv in enumerate(self.convs):
            if k < len(A_powers):
                Ax = torch.einsum("jk,bkc->bjc", A_powers[k], x_flat)
                Ax = Ax.reshape(B, T, J, C)
            else:
                Ax = x

            parts.append(conv(Ax))

        out = torch.cat(parts, dim=-1)  # (B, T, J, C_out)

        # Residual
        res = x
        B2, T2, J2, C2 = res.shape
        res_flat = res.reshape(B2 * T2 * J2, C2)
        if isinstance(self.residual, nn.Sequential):
            res_flat = self.residual[0](res_flat)
            res_flat = self.residual[1](res_flat)
        res = res_flat.reshape(B2, T2, J2, -1)

        out = F.relu(out + res, inplace=True)
        return self.dropout(out)


class TemporalConv(nn.Module):
    """
    1D temporal convolution applied across the time dimension.
    Input: (B, T, J, C) -> Output: (B, T', J, C_out) with optional stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, C)
        Returns:
            out: (B, T', J, C_out)
        """
        B, T, J, C = x.shape
        # Merge B*J, apply temporal conv, restore
        x = x.permute(0, 2, 3, 1).reshape(B * J, C, T)  # (B*J, C, T)
        x = self.conv(x)                                   # (B*J, C_out, T')
        x = self.bn(x)
        T2 = x.shape[-1]
        x = x.reshape(B, J, -1, T2).permute(0, 3, 1, 2)  # (B, T', J, C_out)
        return F.relu(self.dropout(x), inplace=True)


class MSGCNEncoder(nn.Module):
    """
    Full Multi-Scale Spatio-Temporal GCN Encoder.

    Produces hierarchical skeleton features:
        S1: (B, T,   J, 128)   temporal stride 1
        S2: (B, T/2, J, 256)   temporal stride 2
        S3: (B, T/4, J, 512)   temporal stride 2

    Global-average-pooled over joints:
        S1_pool: (B, T,   128)
        S2_pool: (B, T/2, 256)
        S3_pool: (B, T/4, 512)
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: list = None,
        num_joints: int = 75,
        num_scales: int = 3,
        adaptive: bool = True,
        dropout: float = 0.25,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [128, 256, 512]

        self.num_joints = num_joints
        self.num_scales = num_scales

        # Pre-compute adjacency powers (A^1, A^2, A^3)
        A = torch.from_numpy(build_adjacency_matrix(num_joints))
        A_powers = [A]
        for _ in range(num_scales - 1):
            A_powers.append(torch.mm(A_powers[-1], A))
        # Normalize each power
        A_powers_norm = []
        for Ak in A_powers:
            d = Ak.sum(dim=1, keepdim=True).clamp(min=1e-6) ** -0.5
            A_powers_norm.append(Ak * d * d.t())
        self.register_buffer("A_powers", torch.stack(A_powers_norm))  # (K, J, J)

        # Input projection
        c0 = hidden_channels[0]
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, c0),
            nn.BatchNorm1d(c0),
            nn.ReLU(inplace=True),
        )

        # Stage 1: stride 1 (output S1 at T)
        c1 = hidden_channels[0]
        self.ms_gcn_1 = MultiScaleGraphConv(c0, c1, num_joints, num_scales, adaptive, dropout)
        self.temp_1 = TemporalConv(c1, c1, kernel_size=3, stride=1, dropout=dropout)

        # Stage 2: stride 2 (output S2 at T/2)
        c2 = hidden_channels[1]
        self.ms_gcn_2 = MultiScaleGraphConv(c1, c2, num_joints, num_scales, adaptive, dropout)
        self.temp_2 = TemporalConv(c2, c2, kernel_size=3, stride=2, dropout=dropout)

        # Stage 3: stride 2 (output S3 at T/4)
        c3 = hidden_channels[2]
        self.ms_gcn_3 = MultiScaleGraphConv(c2, c3, num_joints, num_scales, adaptive, dropout)
        self.temp_3 = TemporalConv(c3, c3, kernel_size=3, stride=2, dropout=dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: skeleton tensor (B, T, J, C_in)  e.g. (B, 16, 75, 4)

        Returns:
            S1: (B, T,   J, 128)
            S2: (B, T/2, J, 256)
            S3: (B, T/4, J, 512)
            S1_pool: (B, T,   128)   joint-average-pooled
            S2_pool: (B, T/2, 256)
            S3_pool: (B, T/4, 512)
        """
        B, T, J, C = x.shape

        # Input projection: apply over (B*T*J, C)
        x_flat = x.reshape(B * T * J, C)
        x_proj = self.input_proj[0](x_flat)
        x_proj = self.input_proj[1](x_proj)
        x_proj = self.input_proj[2](x_proj)
        x = x_proj.reshape(B, T, J, -1)

        A_list = [self.A_powers[k] for k in range(self.num_scales)]

        # Stage 1
        x = self.ms_gcn_1(x, A_list)  # (B, T, J, 128)
        S1 = x
        x = self.temp_1(x)             # (B, T, J, 128)

        # Stage 2
        x = self.ms_gcn_2(x, A_list)  # (B, T, J, 256)
        x = self.temp_2(x)             # (B, T/2, J, 256)
        S2 = x

        # Stage 3
        x = self.ms_gcn_3(x, A_list)  # (B, T/2, J, 512)
        x = self.temp_3(x)             # (B, T/4, J, 512)
        S3 = x

        # Global joint average pooling
        S1_pool = S1.mean(dim=2)   # (B, T, 128)
        S2_pool = S2.mean(dim=2)   # (B, T/2, 256)
        S3_pool = S3.mean(dim=2)   # (B, T/4, 512)

        return S1, S2, S3, S1_pool, S2_pool, S3_pool
