"""Tests for image analysis module."""

import torch
import numpy as np
import pytest
from PIL import Image
from io import BytesIO

from modules.image_analysis.model import build_model
from modules.image_analysis.preprocess import EVAL_TRANSFORMS, load_image_pil
from config import LEUKEMIA_CLASSES, NUM_CLASSES


def _make_dummy_image(size=(224, 224)) -> Image.Image:
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_model_output_shape():
    model = build_model(pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits, features = model(dummy)
    assert logits.shape == (2, NUM_CLASSES)
    assert features.shape[0] == 2


def test_eval_transforms():
    img = _make_dummy_image()
    tensor = EVAL_TRANSFORMS(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_class_count():
    assert len(LEUKEMIA_CLASSES) == NUM_CLASSES
