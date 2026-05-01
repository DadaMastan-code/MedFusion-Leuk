"""Confusion matrix with heatmap visualization."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    save_path: Path | str | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"{title}" + (" (Normalised)" if normalize else ""), fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    import xgboost as xgb
    import matplotlib
    matplotlib.use("Agg")

    FEATURE_DIR = Path("data/processed/features")
    MODEL_DIR   = Path("models/xgboost")
    PLOT_DIR    = Path("evaluation/plots")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    X_te  = np.load(FEATURE_DIR / "fusion_features_test.npy")
    y_te  = np.load(FEATURE_DIR / "fusion_labels_test.npy")
    classes = list(np.load(FEATURE_DIR / "classes.npy"))

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "fusion_classifier.json"))
    y_pred = model.predict(X_te)

    fig = plot_confusion_matrix(y_te, y_pred, classes=classes,
                                save_path=PLOT_DIR / "confusion_matrix.png",
                                normalize=True, title="MedFusion-Leuk — Fusion Classifier")
    plt.close(fig)

    # Also save raw counts version
    fig2 = plot_confusion_matrix(y_te, y_pred, classes=classes,
                                 save_path=PLOT_DIR / "confusion_matrix_counts.png",
                                 normalize=False, title="MedFusion-Leuk — Fusion Classifier (counts)")
    plt.close(fig2)

    from sklearn.metrics import classification_report
    print(f"Classes: {classes}")
    print(classification_report(y_te, y_pred, target_names=classes, digits=4))
    print(f"Saved: {PLOT_DIR / 'confusion_matrix.png'}")
    print(f"Saved: {PLOT_DIR / 'confusion_matrix_counts.png'}")
