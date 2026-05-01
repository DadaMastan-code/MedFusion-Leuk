"""
Training script for EfficientNet-ViT image model.
Run: python notebooks/01_train_image_model.py

Expects data in data/raw/kaggle_leukemia/ with class subdirectories:
  Normal / ALL / AML / CLL / CML
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
from tqdm import tqdm

from config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_WORKERS, IMAGE_MODEL_DIR, NUM_CLASSES
)
from modules.image_analysis.model import build_model
from modules.image_analysis.preprocess import TRAIN_TRANSFORMS, EVAL_TRANSFORMS

DATA_DIR = Path("data/raw/kaggle_leukemia")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    train_ds = ImageFolder(DATA_DIR / "train", transform=TRAIN_TRANSFORMS)
    val_ds = ImageFolder(DATA_DIR / "val", transform=EVAL_TRANSFORMS)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, correct, n = 0, 0, 0
        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)

        train_acc = correct / n
        val_acc = _evaluate(model, val_dl)
        scheduler.step()
        print(f"Epoch {epoch}: loss={total_loss/n:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            IMAGE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), IMAGE_MODEL_DIR / "best_model.pt")
            print(f"  → Saved best model (val_acc={val_acc:.4f})")


@torch.no_grad()
def _evaluate(model, loader) -> float:
    model.eval()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits, _ = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return correct / n


if __name__ == "__main__":
    print(f"Training on {DEVICE}")
    train()
