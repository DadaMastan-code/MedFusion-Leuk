"""
Evaluation metrics: accuracy, precision, recall, F1, AUC-ROC, cross-validation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from sklearn.model_selection import StratifiedKFold
from config import LEUKEMIA_CLASSES, CV_FOLDS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return a dict of all key metrics for a set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "classification_report": classification_report(
            y_true, y_pred, target_names=LEUKEMIA_CLASSES, zero_division=0
        ),
    }


def cross_validate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold stratified cross-validation; returns mean ± std for key metrics."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "f1_macro": [], "roc_auc_ovr": []}

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        import copy
        fold_model = copy.deepcopy(model)
        fold_model.fit(X_tr, y_tr, feature_names=[f"f{i}" for i in range(X_tr.shape[1])])
        _, proba = fold_model.predict(X_val)
        preds = proba.argmax(axis=1)

        fold_metrics["accuracy"].append(accuracy_score(y_val, preds))
        fold_metrics["f1_macro"].append(f1_score(y_val, preds, average="macro", zero_division=0))
        fold_metrics["roc_auc_ovr"].append(
            roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
        )

    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in fold_metrics.items()
    }
