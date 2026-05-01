"""
Comparative analysis table: MedFusion-Leuk vs standalone baselines.
Uses the actual pre-split train / test data from data/processed/features/.

Baselines:
  1. SVM              — full 448-dim fused features, SVC(rbf, C=10)
  2. Random Forest     — full 448-dim fused features, n_estimators=200
  3. CNN only          — image features only (first 256 dims)
  4. NLP only          — NLP feature vector only (last 128 dims), XGBoost
  5. MedFusion-Leuk    — full 448-dim, XGBoost (all modalities)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

BASELINE_NAMES = [
    "SVM (full features)",
    "Random Forest (full features)",
    "CNN only (image, 256-dim)",
    "NLP only (keyword, 128-dim)",
    "MedFusion-Leuk (all modalities)",
]


def run_comparative(
    X_full: np.ndarray,
    y: np.ndarray,
    X_image: np.ndarray,
    X_nlp: np.ndarray,
    medfusion_proba: np.ndarray,
    save_path: Path | str | None = None,
) -> pd.DataFrame:
    """Library-compatible entry point (used by train_fusion_classifier.py)."""
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)
    xi_tr = X_image[:len(X_tr)]; xi_te = X_image[len(X_tr):]
    xn_tr = X_nlp[:len(X_tr)];   xn_te = X_nlp[len(X_tr):]

    rows = []
    for name, xtr, xte in [
        ("SVM (full features)",            X_tr,  X_te),
        ("Random Forest (full features)",  X_tr,  X_te),
        ("CNN only (image, 256-dim)",       xi_tr, xi_te),
        ("NLP only (keyword, 128-dim)",     xn_tr, xn_te),
    ]:
        clf = (SVC(probability=True, kernel="rbf", random_state=42)
               if "SVM" in name
               else RandomForestClassifier(n_estimators=200, random_state=42))
        clf.fit(np.nan_to_num(xtr), y_tr)
        p = clf.predict_proba(np.nan_to_num(xte))
        rows.append(_eval_row(name, y_te, p))

    rows.append(_eval_row("MedFusion-Leuk (all modalities)", y_te,
                          medfusion_proba[-len(y_te):]))

    df = pd.DataFrame(rows).set_index("Method")
    if save_path:
        _plot_comparison(df, save_path)
    return df


def _eval_row(name: str, y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    preds = y_proba.argmax(axis=1)
    nc = y_proba.shape[1]
    auc = (roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
           if nc > 2 else roc_auc_score(y_true, y_proba[:, 1]))
    return {
        "Method":    name,
        "Accuracy":  round(accuracy_score(y_true, preds), 4),
        "Precision": round(precision_score(y_true, preds, average="macro", zero_division=0), 4),
        "Recall":    round(recall_score(y_true, preds, average="macro", zero_division=0), 4),
        "F1 Macro":  round(f1_score(y_true, preds, average="macro", zero_division=0), 4),
        "AUC-ROC":   round(auc, 4),
    }


def _plot_comparison(df: pd.DataFrame, save_path: Path | str) -> None:
    metrics = ["Accuracy", "F1 Macro", "AUC-ROC"]
    n = len(df)
    colors = ["#95a5a6"] * (n - 1) + ["#e74c3c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, metrics):
        ax.barh(df.index, df[col], color=colors)
        ax.set_xlim(max(0, df[col].min() - 0.06), 1.02)
        ax.set_title(col, fontsize=12)
        ax.invert_yaxis()
        for i, v in enumerate(df[col]):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    plt.suptitle("Comparative Analysis: MedFusion-Leuk vs Baselines (C-NMC 2019)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import xgboost as xgb

    FEATURE_DIR = Path("data/processed/features")
    MODEL_DIR   = Path("models/xgboost")
    EVAL_DIR    = Path("evaluation")
    PLOT_DIR    = EVAL_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Load actual pre-split data
    X_tr  = np.load(FEATURE_DIR / "fusion_features_train.npy")
    y_tr  = np.load(FEATURE_DIR / "fusion_labels_train.npy")
    X_va  = np.load(FEATURE_DIR / "fusion_features_val.npy")
    y_va  = np.load(FEATURE_DIR / "fusion_labels_val.npy")
    X_te  = np.load(FEATURE_DIR / "fusion_features_test.npy")
    y_te  = np.load(FEATURE_DIR / "fusion_labels_test.npy")
    classes = list(np.load(FEATURE_DIR / "classes.npy"))

    # Combine train+val for baseline fitting (same as XGBoost training)
    X_trainval = np.vstack([X_tr, X_va])
    y_trainval = np.concatenate([y_tr, y_va])

    # Feature slices
    IMG_DIM, PDF_DIM, NLP_DIM = 256, 64, 128
    X_img_tr = X_trainval[:, :IMG_DIM]
    X_nlp_tr = X_trainval[:, IMG_DIM + PDF_DIM:]
    X_img_te = X_te[:, :IMG_DIM]
    X_nlp_te = X_te[:, IMG_DIM + PDF_DIM:]

    rows = []

    # ── Baseline 1: SVM on full 448-dim fused features ─────────────────────
    print("Training Baseline 1 — SVM (full 448-dim, C=10)...")
    svm = SVC(probability=True, kernel="rbf", C=10, random_state=42)
    svm.fit(np.nan_to_num(X_trainval), y_trainval)
    p = svm.predict_proba(np.nan_to_num(X_te))
    rows.append(_eval_row("SVM (full features)", y_te, p))
    print(f"  SVM: Acc={rows[-1]['Accuracy']:.4f}  F1={rows[-1]['F1 Macro']:.4f}  AUC={rows[-1]['AUC-ROC']:.4f}")

    # ── Baseline 2: Random Forest on full 448-dim fused features ───────────
    print("Training Baseline 2 — Random Forest (full 448-dim, n=200)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(np.nan_to_num(X_trainval), y_trainval)
    p = rf.predict_proba(np.nan_to_num(X_te))
    rows.append(_eval_row("Random Forest (full features)", y_te, p))
    print(f"  RF:  Acc={rows[-1]['Accuracy']:.4f}  F1={rows[-1]['F1 Macro']:.4f}  AUC={rows[-1]['AUC-ROC']:.4f}")

    # ── Baseline 3: CNN only — image features (256-dim) ────────────────────
    print("Training Baseline 3 — CNN only (image features, 256-dim)...")
    rf_img = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_img.fit(np.nan_to_num(X_img_tr), y_trainval)
    p = rf_img.predict_proba(np.nan_to_num(X_img_te))
    rows.append(_eval_row("CNN only (image, 256-dim)", y_te, p))
    print(f"  CNN: Acc={rows[-1]['Accuracy']:.4f}  F1={rows[-1]['F1 Macro']:.4f}  AUC={rows[-1]['AUC-ROC']:.4f}")

    # ── Baseline 4: NLP only — keyword features (128-dim), XGBoost ─────────
    print("Training Baseline 4 — NLP only (keyword features, 128-dim, XGBoost)...")
    nlp_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, verbosity=0, random_state=42,
    )
    nlp_xgb.fit(np.nan_to_num(X_nlp_tr), y_trainval)
    p = nlp_xgb.predict_proba(np.nan_to_num(X_nlp_te))
    rows.append(_eval_row("NLP only (keyword, 128-dim)", y_te, p))
    print(f"  NLP: Acc={rows[-1]['Accuracy']:.4f}  F1={rows[-1]['F1 Macro']:.4f}  AUC={rows[-1]['AUC-ROC']:.4f}")

    # ── Baseline 5: MedFusion-Leuk (full 448-dim XGBoost) ──────────────────
    print("Evaluating MedFusion-Leuk — full 448-dim XGBoost...")
    fusion_model = xgb.XGBClassifier()
    fusion_model.load_model(str(MODEL_DIR / "fusion_classifier.json"))
    p = fusion_model.predict_proba(np.nan_to_num(X_te))
    rows.append(_eval_row("MedFusion-Leuk (all modalities)", y_te, p))
    print(f"  MedFusion: Acc={rows[-1]['Accuracy']:.4f}  F1={rows[-1]['F1 Macro']:.4f}  AUC={rows[-1]['AUC-ROC']:.4f}")

    # ── Print table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows).set_index("Method")

    print("\n" + "=" * 85)
    print(f"{'Model':<38} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'AUC':>6}")
    print("-" * 85)
    for method, row in df.iterrows():
        marker = " <--" if "MedFusion" in method else ""
        print(f"  {method:<36} | {row['Accuracy']:>8.4f} | {row['Precision']:>9.4f} | "
              f"{row['Recall']:>6.4f} | {row['F1 Macro']:>6.4f} | {row['AUC-ROC']:>6.4f}{marker}")
    print("=" * 85)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = EVAL_DIR / "comparative_results.csv"
    df.to_csv(csv_path)
    print(f"\nCSV saved:   {csv_path}")

    # ── Save bar chart ───────────────────────────────────────────────────────
    _plot_comparison(df, PLOT_DIR / "comparative_analysis.png")
    print(f"Chart saved: {PLOT_DIR / 'comparative_analysis.png'}")
