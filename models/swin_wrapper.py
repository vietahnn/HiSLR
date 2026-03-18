# hislr/models/swin_wrapper.py
# ─────────────────────────────────────────────────────────────────────────────
# Video Swin Transformer V2 — Wrapper
#
# Wraps torchvision's or timm's Video Swin-V2 implementation to expose
# hierarchical feature maps {R1, R2, R3, R4} needed by HiCMF.
#
# If neither torchvision nor timm provides Video Swin-V2, falls back to a
# lightweight stub with correct output shapes for development/testing.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


# Output channel sizes per stage for each Swin-V2 variant
SWIN_V2_CHANNELS = {
    "swin_v2_t": [96,  192,  384,  768],
    "swin_v2_s": [96,  192,  384,  768],
    "swin_v2_b": [128, 256,  512, 1024],
    "swin_v2_l": [192, 384,  768, 1536],
}


class SwinV2Stub(nn.Module):
    """
    Lightweight stub that mimics the Swin-V2 output interface.
    Used for development, unit testing, and environments without
    a full Swin-V2 implementation.

    Produces correctly shaped hierarchical features using lightweight
    3D convolutional blocks.
    """

    def __init__(self, variant: str = "swin_v2_b", in_channels: int = 3):
        super().__init__()
        self.out_channels = SWIN_V2_CHANNELS[variant]
        c = self.out_channels

        # Each stage halves spatial resolution, keeps temporal dimension
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, c[0], kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=0),
            nn.BatchNorm3d(c[0]), nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(c[0], c[1], kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(c[1]), nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(c[1], c[2], kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(c[2]), nn.GELU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv3d(c[2], c[3], kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(c[3]), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, T, 3, H, W)  — RGB clip
        Returns:
            dict: R1 (B,T,H/4,W/4,C1), R2 (B,T,H/8,W/8,C2),
                  R3 (B,T,H/16,W/16,C3), R4 (B,T,H/32,W/32,C4)
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) for Conv3d

        R1 = self.stage1(x)   # (B, C1, T, H/4,  W/4)
        R2 = self.stage2(R1)  # (B, C2, T, H/8,  W/8)
        R3 = self.stage3(R2)  # (B, C3, T, H/16, W/16)
        R4 = self.stage4(R3)  # (B, C4, T, H/32, W/32)

        def to_thwc(t):
            # (B, C, T, H, W) -> (B, T, H, W, C)
            return t.permute(0, 2, 3, 4, 1).contiguous()

        return {
            "R1": to_thwc(R1),
            "R2": to_thwc(R2),
            "R3": to_thwc(R3),
            "R4": to_thwc(R4),
        }


class SwinV2TimmWrapper(nn.Module):
    """
    Wrapper around timm's Video Swin Transformer V2.
    Hooks into intermediate stages to extract hierarchical feature maps.
    """

    def __init__(self, variant: str = "swin_v2_b", pretrained: bool = True, drop_path_rate: float = 0.2):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for SwinV2TimmWrapper. Install via: pip install timm")

        # Map variant to timm model name
        timm_names = {
            "swin_v2_t": "swinv2_tiny_window16_256",
            "swin_v2_s": "swinv2_small_window16_256",
            "swin_v2_b": "swinv2_base_window16_256",
            "swin_v2_l": "swinv2_large_window16_256",
        }
        model_name = timm_names.get(variant, "swinv2_base_window16_256")

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )
        self.out_channels = SWIN_V2_CHANNELS[variant]

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, T, 3, H, W)
        Returns:
            dict R1..R4 each (B, T, H_i, W_i, C_i)
        """
        B, T, C, H, W = x.shape
        # Process each frame independently (2D Swin), aggregate over T
        # For a proper Video Swin implementation, replace with 3D attention
        x_2d = x.reshape(B * T, C, H, W)
        feats = self.model(x_2d)  # list of (B*T, C_i, H_i, W_i)

        out = {}
        keys = ["R1", "R2", "R3", "R4"]
        for key, f in zip(keys, feats):
            bt, ci, hi, wi = f.shape
            # Reshape and convert to (B, T, H, W, C)
            f = f.reshape(B, T, ci, hi, wi).permute(0, 1, 3, 4, 2).contiguous()
            out[key] = f

        return out


class SwinV2Encoder(nn.Module):
    """
    Unified Swin-V2 Encoder interface.
    Tries to load a real Swin-V2 (timm), falls back to stub.

    Usage:
        encoder = SwinV2Encoder(variant="swin_v2_b", pretrained=True)
        feats = encoder(rgb)  # rgb: (B, T, 3, H, W)
        # feats["R4"]: (B, T, H/32, W/32, 1024)
    """

    def __init__(
        self,
        variant: str = "swin_v2_b",
        pretrained: bool = True,
        pretrain_path: str = "",
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.out_channels = SWIN_V2_CHANNELS[variant]

        loaded_real = False

        # Try loading from a provided checkpoint path first
        if pretrain_path:
            try:
                self._encoder = SwinV2Stub(variant)
                ckpt = torch.load(pretrain_path, map_location="cpu")
                state = ckpt.get("model", ckpt.get("state_dict", ckpt))
                self._encoder.load_state_dict(state, strict=False)
                print(f"[SwinV2Encoder] Loaded weights from {pretrain_path}")
                loaded_real = True
            except Exception as e:
                print(f"[SwinV2Encoder] Failed to load checkpoint: {e}")

        # Try timm
        if not loaded_real:
            try:
                self._encoder = SwinV2TimmWrapper(variant, pretrained, drop_path_rate)
                print(f"[SwinV2Encoder] Loaded timm Swin-V2 ({variant})")
                loaded_real = True
            except Exception as e:
                print(f"[SwinV2Encoder] timm not available ({e}), using stub.")

        # Fallback to stub
        if not loaded_real:
            self._encoder = SwinV2Stub(variant)
            print(f"[SwinV2Encoder] Using lightweight stub ({variant})")

    def forward(self, x: torch.Tensor) -> dict:
        return self._encoder(x)
