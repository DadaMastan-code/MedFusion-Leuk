"""
Training script for the XGBoost fusion classifier.
Requires trained image model + labelled dataset with images and blood reports.

Run: python notebooks/02_train_fusion_classifier.py
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import LEUKEMIA_CLASSES, IMAGE_FEATURE_SIZE, PDF_FEATURE_SIZE, NLP_FEATURE_SIZE
from modules.image_analysis.model import build_model
from modules.image_analysis.preprocess import EVAL_TRANSFORMS
from modules.pdf_extraction.parser import parse_report
from modules.nlp.biobert_ner import BioBERTNER
from modules.fusion.multimodal_fusion import AttentionFusionLayer, MultimodalFusion
from modules.prediction.classifier import LeukemiaClassifier
from evaluation.metrics import compute_metrics, cross_validate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/processed")


def build_feature_matrix():
    """
    Placeholder: load pre-computed feature vectors.
    In a real run: iterate over the dataset, run each module, and collect vectors.
    Replace with actual data loading logic.
    """
    print("[WARNING] Using synthetic data — replace with real dataset loading.")
    n_samples = 500
    n_classes = len(LEUKEMIA_CLASSES)
    X = np.random.randn(n_samples, 256).astype(np.float32)  # 256 = COMMON_DIM from fusion
    y = np.random.randint(0, n_classes, n_samples)
    feature_names = [f"fused_dim_{i}" for i in range(256)]
    return X, y, feature_names


def train():
    X, y, feature_names = build_feature_matrix()

    print(f"Training XGBoost classifier on {len(X)} samples, {len(LEUKEMIA_CLASSES)} classes …")
    clf = LeukemiaClassifier()
    clf.fit(X, y, feature_names)

    preds, proba = clf.predict(X)
    metrics = compute_metrics(y, preds, proba)
    print("\nTraining metrics (train set — use cross-validation for unbiased estimates):")
    for k, v in metrics.items():
        if k != "classification_report":
            print(f"  {k}: {v:.4f}")
    print("\nClassification Report:\n", metrics["classification_report"])

    print("\nRunning 5-fold cross-validation …")
    cv_results = cross_validate(clf, X, y)
    for metric, stats in cv_results.items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    model_path = clf.save()
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
