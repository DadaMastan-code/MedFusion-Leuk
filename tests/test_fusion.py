"""Tests for multimodal fusion layer."""

import numpy as np
import pytest
import torch
from modules.fusion.multimodal_fusion import AttentionFusionLayer, MultimodalFusion
from config import IMAGE_FEATURE_SIZE, PDF_FEATURE_SIZE, NLP_FEATURE_SIZE


def _fusion():
    layer = AttentionFusionLayer()
    return MultimodalFusion(layer)


def test_all_modalities():
    fusion = _fusion()
    img = np.random.randn(IMAGE_FEATURE_SIZE).astype(np.float32)
    pdf = np.random.randn(PDF_FEATURE_SIZE).astype(np.float32)
    nlp = np.random.randn(NLP_FEATURE_SIZE).astype(np.float32)
    out = fusion.fuse(img, pdf, nlp)
    assert out.shape == (AttentionFusionLayer.COMMON_DIM,)


def test_image_only():
    fusion = _fusion()
    img = np.random.randn(IMAGE_FEATURE_SIZE).astype(np.float32)
    out = fusion.fuse(img_vec=img)
    assert out.shape == (AttentionFusionLayer.COMMON_DIM,)


def test_pdf_only():
    fusion = _fusion()
    pdf = np.random.randn(PDF_FEATURE_SIZE).astype(np.float32)
    out = fusion.fuse(pdf_vec=pdf)
    assert out.shape == (AttentionFusionLayer.COMMON_DIM,)


def test_no_modality_raises():
    fusion = _fusion()
    with pytest.raises(ValueError):
        fusion.fuse()


def test_nan_imputation():
    """NaN values in a provided vector must not cause errors."""
    fusion = _fusion()
    img = np.full(IMAGE_FEATURE_SIZE, np.nan, dtype=np.float32)
    out = fusion.fuse(img_vec=img)
    assert not np.isnan(out).any()
