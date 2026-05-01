"""
Step 3 — Train EfficientNet-B4 + ViT hybrid on leukemia image dataset.
Reads split CSVs produced by prepare_dataset.py.
Adapts automatically to 2-class or 5-class datasets.
"""

import sys, time, warnings
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.image_analysis.model import build_model
from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, IMAGE_MODEL_DIR

SPLIT_DIR  = Path("data/processed/splits")
EVAL_DIR   = Path("evaluation/plots")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 32 if DEVICE.type == "cuda" else (16 if DEVICE.type == "mps" else 8)
MAX_EPOCHS = 30 if DEVICE.type == "cuda" else 15
PATIENCE   = 5
LR, WD     = 1e-4, 1e-4

print(f"Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}")


# ── Dataset ────────────────────────────────────────────────────────────────
class LeukemiaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.classes.index(row.label)
        return img, label


TRAIN_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
])


def load_splits():
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    val_df   = pd.read_csv(SPLIT_DIR / "val.csv")
    test_df  = pd.read_csv(SPLIT_DIR / "test.csv")
    classes  = sorted(train_df.label.unique().tolist())
    print(f"\nClasses detected: {classes}")
    return train_df, val_df, test_df, classes


def compute_weights(train_df, classes):
    labels = [classes.index(l) for l in train_df.label]
    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum = correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct  += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return loss_sum / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum = correct = n = 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * len(labels)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return loss_sum / n, correct / n, all_preds, all_labels


def save_confusion_matrix(y_true, y_pred, classes, path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, len(classes)*1.5), max(5, len(classes)*1.3)))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Image Model — Confusion Matrix (Normalised)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {path}")


def main():
    train_df, val_df, test_df, classes = load_splits()
    num_classes = len(classes)

    train_ds = LeukemiaDataset(train_df, classes, TRAIN_TF)
    val_ds   = LeukemiaDataset(val_df,   classes, EVAL_TF)
    test_ds  = LeukemiaDataset(test_df,  classes, EVAL_TF)

    nw = 0  # MPS and CPU both work best with 0 workers on macOS
    pin = DEVICE.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin)

    print(f"\nDataset sizes — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    model     = build_model(num_classes=num_classes, pretrained=True).to(DEVICE)
    weights   = compute_weights(train_df, classes)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_acc = 0.0
    epochs_no_improve = 0
    history = []

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | Best")
    print("-" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_dl, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = vl_acc > best_val_acc
        marker = " <-- best" if is_best else ""
        print(f"{epoch:>5} | {tr_loss:>10.4f} | {tr_acc:>9.4f} | {vl_loss:>9.4f} | {vl_acc:>8.4f}{marker}  ({elapsed:.0f}s)")

        if is_best:
            best_val_acc = vl_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": vl_acc,
                "classes": classes,
            }, IMAGE_MODEL_DIR / "best_model.pth")
        else:
            epochs_no_improve += 1

        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "vl_loss": vl_loss, "vl_acc": vl_acc})

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # ── Test evaluation ────────────────────────────────────────────────────
    print(f"\n=== TEST SET EVALUATION ===")
    ckpt = torch.load(IMAGE_MODEL_DIR / "best_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_dl, criterion)

    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print("\nPer-class metrics:")
    print(classification_report(test_labels, test_preds, target_names=classes, digits=4))

    save_confusion_matrix(
        test_labels, test_preds, classes,
        EVAL_DIR / "image_confusion_matrix.png"
    )

    # ── Save training curves ───────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist_df.epoch, hist_df.tr_loss, label="Train"); ax1.plot(hist_df.epoch, hist_df.vl_loss, label="Val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(hist_df.epoch, hist_df.tr_acc, label="Train"); ax2.plot(hist_df.epoch, hist_df.vl_acc, label="Val")
    ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout()
    fig.savefig(EVAL_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Training curves saved: {EVAL_DIR / 'training_curves.png'}")

    return test_acc, classes


import os
if __name__ == "__main__":
    main()
