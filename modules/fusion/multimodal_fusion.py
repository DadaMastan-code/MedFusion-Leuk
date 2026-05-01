"""
Attention-weighted late fusion of image, PDF, and NLP feature vectors.

# [NOVELTY] Graceful modality degradation: the attention gate is re-normalised
# over only the available modalities, so the system operates coherently when
# one or two inputs are missing — clinically realistic in resource-limited settings.

# [NOVELTY] Context-aware multimodal arbitration (apply_parameter_confidence):
# when quantitative lab parameters (blast_pct, WBC) are available and indicate
# normal values, the NLP leukemia keyword flag is dampened to prevent text-driven
# false positives. This implements "parameter-confidence arbitration" — quantitative
# evidence overrides ambiguous textual keyword cues, a novel design decision that
# prevents clinically dangerous false positives in keyword-text vs. lab-value conflicts.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from config import IMAGE_FEATURE_SIZE, PDF_FEATURE_SIZE, NLP_FEATURE_SIZE, FUSED_FEATURE_SIZE


class ModalityProjector(nn.Module):
    """Projects each modality into a common embedding space of `out_dim`."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionFusionLayer(nn.Module):
    """
    # [NOVELTY] Attention-weighted multimodal fusion.
    # F_fused = Σ(αᵢ · Fᵢ)  where αᵢ are softmax-normalised learned weights
    # restricted to the set of available (non-missing) modalities.
    """

    COMMON_DIM = 256

    def __init__(self):
        super().__init__()
        self.img_proj = ModalityProjector(IMAGE_FEATURE_SIZE, self.COMMON_DIM)
        self.pdf_proj = ModalityProjector(PDF_FEATURE_SIZE, self.COMMON_DIM)
        self.nlp_proj = ModalityProjector(NLP_FEATURE_SIZE, self.COMMON_DIM)

        # Attention scorer: maps each projected vector to a scalar logit
        self.attn_scorer = nn.Linear(self.COMMON_DIM, 1, bias=False)

    def forward(
        self,
        img_feat: Optional[torch.Tensor],   # (B, IMAGE_FEATURE_SIZE) or None
        pdf_feat: Optional[torch.Tensor],   # (B, PDF_FEATURE_SIZE) or None
        nlp_feat: Optional[torch.Tensor],   # (B, NLP_FEATURE_SIZE) or None
    ) -> torch.Tensor:
        """Returns fused feature tensor (B, COMMON_DIM)."""
        projected = []
        if img_feat is not None:
            projected.append(self.img_proj(img_feat))
        if pdf_feat is not None:
            projected.append(self.pdf_proj(pdf_feat))
        if nlp_feat is not None:
            projected.append(self.nlp_proj(nlp_feat))

        if not projected:
            raise ValueError("At least one modality must be provided.")

        # Stack → (B, M, COMMON_DIM) where M = number of available modalities
        stacked = torch.stack(projected, dim=1)

        # Scalar attention logit per modality
        logits = self.attn_scorer(stacked).squeeze(-1)  # (B, M)
        weights = torch.softmax(logits, dim=-1)          # (B, M)

        # Weighted sum
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B, COMMON_DIM)
        return fused


# ── Numpy convenience wrapper (inference without gradient) ─────────────────

class MultimodalFusion:
    """
    Stateless numpy wrapper used during inference.
    Accepts raw float32 arrays, handles NaN imputation, returns fused vector.
    """

    def __init__(self, model: AttentionFusionLayer):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def fuse(
        self,
        img_vec: Optional[np.ndarray] = None,
        pdf_vec: Optional[np.ndarray] = None,
        nlp_vec: Optional[np.ndarray] = None,
        blast_pct: Optional[float] = None,
        wbc: Optional[float] = None,
    ) -> np.ndarray:
        """
        Each vector should be float32 of the declared size.
        Missing modalities should be passed as None.
        NaN values within a provided vector are replaced with 0.
        blast_pct / wbc: when both are available and indicate normal values,
        the NLP vector is rescaled via apply_parameter_confidence before fusion.
        """
        if nlp_vec is not None:
            nlp_vec = apply_parameter_confidence(nlp_vec, blast_pct=blast_pct, wbc=wbc)

        def _to_tensor(vec: np.ndarray) -> torch.Tensor:
            arr = np.nan_to_num(vec, nan=0.0)
            return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, D)

        img_t = _to_tensor(img_vec) if img_vec is not None else None
        pdf_t = _to_tensor(pdf_vec) if pdf_vec is not None else None
        nlp_t = _to_tensor(nlp_vec) if nlp_vec is not None else None

        fused = self.model(img_t, pdf_t, nlp_t)  # (1, COMMON_DIM)
        return fused.squeeze(0).numpy()


def apply_parameter_confidence(
    nlp_feat: np.ndarray,
    blast_pct: Optional[float] = None,
    wbc: Optional[float] = None,
) -> np.ndarray:
    """
    # [NOVELTY] Context-aware multimodal arbitration — when quantitative lab
    # parameters are available and indicate a normal blast count, the NLP
    # leukemia keyword flag (index 12) is dampened proportionally to the
    # parameter confidence score. This prevents text keyword false positives
    # when the actual measured values contradict the textual cues.
    #
    # parameter_confidence = 0.6 (blast_pct available) + 0.4 (wbc available)
    # If blast_pct <= 5 and WBC is not hyperleukocytotic (< 15,000): scale flag down.
    """
    if blast_pct is None and wbc is None:
        return nlp_feat

    param_confidence = (0.6 if blast_pct is not None else 0.0) + \
                       (0.4 if wbc is not None else 0.0)

    if blast_pct is not None and blast_pct <= 5.0:
        wbc_ok = wbc is None or wbc <= 15_000
        if wbc_ok:
            result = nlp_feat.copy()
            result[12] *= max(0.0, 1.0 - param_confidence)
            return result

    return nlp_feat
