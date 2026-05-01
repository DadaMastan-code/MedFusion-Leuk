"""Image preprocessing for blood smear images."""

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])


def load_image(path: str | Path, train: bool = False) -> torch.Tensor:
    """Load a blood smear image and return a normalised tensor (C,H,W)."""
    img = Image.open(path).convert("RGB")
    tfm = TRAIN_TRANSFORMS if train else EVAL_TRANSFORMS
    return tfm(img)


def load_image_pil(path: str | Path) -> Image.Image:
    """Return original PIL image (for Grad-CAM overlay)."""
    return Image.open(path).convert("RGB")
