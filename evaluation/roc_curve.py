"""ROC-AUC curves per class (One-vs-Rest), handles binary and multi-class."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    if classes is None:
        classes = [str(i) for i in range(y_proba.shape[1])]

    n_classes = len(classes)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[0], lw=2, label=f"{classes[1]} (AUC={roc_auc:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i, (cls, color) in enumerate(zip(classes, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — MedFusion-Leuk (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    import xgboost as xgb
    import matplotlib
    matplotlib.use("Agg")
    from sklearn.metrics import roc_auc_score

    FEATURE_DIR = Path("data/processed/features")
    MODEL_DIR   = Path("models/xgboost")
    PLOT_DIR    = Path("evaluation/plots")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    X_te    = np.load(FEATURE_DIR / "fusion_features_test.npy")
    y_te    = np.load(FEATURE_DIR / "fusion_labels_test.npy")
    classes = list(np.load(FEATURE_DIR / "classes.npy"))

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "fusion_classifier.json"))
    y_proba = model.predict_proba(X_te)

    fig = plot_roc_curves(y_te, y_proba, classes=classes,
                          save_path=PLOT_DIR / "roc_curve.png")
    plt.close(fig)

    auc_score = roc_auc_score(y_te, y_proba[:, 1])
    print(f"Classes:  {classes}")
    print(f"AUC-ROC:  {auc_score:.4f}")
    print(f"Saved: {PLOT_DIR / 'roc_curve.png'}")
