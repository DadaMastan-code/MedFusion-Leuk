"""
XGBoost classifier trained on fused multimodal feature vectors.
Includes SHAP explainability.
"""

import numpy as np
import shap
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from config import (
    LEUKEMIA_CLASSES, XGBOOST_PARAMS, XGBOOST_MODEL_DIR, SHAP_PLOT_DIR
)


class LeukemiaClassifier:
    """XGBoost classifier with built-in SHAP explanation."""

    def __init__(self):
        self.model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: list[str] = []

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (class_indices, probabilities)."""
        X = np.nan_to_num(X, nan=0.0)
        proba = self.model.predict_proba(X)
        preds = proba.argmax(axis=1)
        return preds, proba

    def predict_single(self, feature_vec: np.ndarray) -> dict:
        """Full prediction dict for one sample."""
        vec = feature_vec.reshape(1, -1)
        preds, proba = self.predict(vec)
        class_idx = int(preds[0])
        confidence = float(proba[0][class_idx])
        return {
            "class_idx": class_idx,
            "predicted_class": LEUKEMIA_CLASSES[class_idx],
            "confidence": confidence,
            "probabilities": {
                cls: float(p) for cls, p in zip(LEUKEMIA_CLASSES, proba[0])
            },
        }

    # ── SHAP explainability ────────────────────────────────────────────────

    def explain(
        self, X: np.ndarray, sample_idx: int = 0, save_path: Optional[Path] = None
    ) -> dict:
        """
        # [NOVELTY] SHAP values show which specific features (WBC, blast %,
        # image embedding dims, NLP entity counts) drove each classification,
        # making the fusion model interpretable to clinicians.
        """
        if self.explainer is None:
            raise RuntimeError("Fit the model before explaining.")

        X = np.nan_to_num(X, nan=0.0)
        shap_values = self.explainer.shap_values(X)

        # shap_values: list of (N, D) arrays, one per class
        sample_shap = shap_values[sample_idx] if isinstance(shap_values, list) \
            else shap_values[sample_idx]

        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Top-10 features by absolute SHAP magnitude
        abs_shap = np.abs(sample_shap).mean(axis=0) if sample_shap.ndim > 1 else np.abs(sample_shap)
        top_idx = abs_shap.argsort()[::-1][:10]
        top_features = {names[i]: float(abs_shap[i]) for i in top_idx}

        if save_path is not None:
            SHAP_PLOT_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            feat_labels = list(top_features.keys())
            feat_vals = list(top_features.values())
            ax.barh(feat_labels[::-1], feat_vals[::-1], color="#e74c3c")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Top Feature Contributions (SHAP)")
            plt.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)

        return {"top_shap_features": top_features}

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (XGBOOST_MODEL_DIR / "leukemia_xgb.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LeukemiaClassifier":
        path = path or (XGBOOST_MODEL_DIR / "leukemia_xgb.pkl")
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.feature_names = data["features"]
        obj.explainer = shap.TreeExplainer(obj.model)
        return obj
