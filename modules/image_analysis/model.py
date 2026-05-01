"""
EfficientNet-B4 + Vision Transformer hybrid for blood smear classification.

# [NOVELTY] Dual-encoder CNN-ViT fusion: EfficientNet captures local cell
# morphology (nuclear contour, chromatin texture) while ViT attends to
# global spatial relationships between cells in the smear.
"""

import torch
import torch.nn as nn
import timm
from config import (
    EFFICIENTNET_VARIANT, VIT_VARIANT, NUM_CLASSES, IMAGE_FEATURE_DIM
)


class EfficientNetViTHybrid(nn.Module):
    """
    Extracts local features via EfficientNet-B4 and global features via ViT,
    fuses them with a learned gating mechanism, then classifies.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # ── Local feature extractor (CNN) ──────────────────────────────────
        self.efficientnet = timm.create_model(
            EFFICIENTNET_VARIANT, pretrained=pretrained, num_classes=0
        )
        cnn_out_dim = self.efficientnet.num_features  # 1792 for B4

        # ── Global feature extractor (ViT) ─────────────────────────────────
        self.vit = timm.create_model(
            VIT_VARIANT, pretrained=pretrained, num_classes=0
        )
        vit_out_dim = self.vit.num_features  # 768 for ViT-B/16

        # ── Projection heads → common embedding space ──────────────────────
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_out_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # ── Learned gating: how much to weight CNN vs ViT ──────────────────
        # [NOVELTY] soft gate instead of fixed concatenation
        self.gate = nn.Sequential(
            nn.Linear(512 * 2, 2),
            nn.Softmax(dim=-1),
        )

        # ── Classifier head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

        # Public hook: stores penultimate-layer activations for Grad-CAM
        self.feature_hook_output: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, num_classes)
            features: (B, 512) fused embedding (used by fusion layer)
        """
        cnn_feat = self.cnn_proj(self.efficientnet(x))   # (B, 512)
        vit_feat = self.vit_proj(self.vit(x))            # (B, 512)

        # Gated fusion
        combined = torch.cat([cnn_feat, vit_feat], dim=-1)  # (B, 1024)
        gates = self.gate(combined)                           # (B, 2)
        fused = gates[:, 0:1] * cnn_feat + gates[:, 1:2] * vit_feat  # (B, 512)

        self.feature_hook_output = fused
        logits = self.classifier(fused)
        return logits, fused

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-d feature vector without classification (for fusion layer)."""
        with torch.no_grad():
            _, features = self.forward(x)
        return features


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> EfficientNetViTHybrid:
    return EfficientNetViTHybrid(num_classes=num_classes, pretrained=pretrained)
