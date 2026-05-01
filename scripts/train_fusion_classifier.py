"""
Step 6 — Train XGBoost fusion classifier on fused feature vectors.
Produces SHAP explanations and a comparative analysis against baselines.
"""

import sys, warnings, os
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

FEATURE_DIR = Path("data/processed/features")
MODEL_DIR   = Path("models/xgboost")
EVAL_DIR    = Path("evaluation/plots")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_tr = np.load(FEATURE_DIR / "fusion_features_train.npy")
    y_tr = np.load(FEATURE_DIR / "fusion_labels_train.npy")
    X_va = np.load(FEATURE_DIR / "fusion_features_val.npy")
    y_va = np.load(FEATURE_DIR / "fusion_labels_val.npy")
    X_te = np.load(FEATURE_DIR / "fusion_features_test.npy")
    y_te = np.load(FEATURE_DIR / "fusion_labels_test.npy")
    classes = list(np.load(FEATURE_DIR / "classes.npy"))
    print(f"Data loaded — Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
    print(f"Classes: {classes}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, classes


def train_xgboost(X_tr, y_tr, X_va, y_va, classes):
    print("\n=== TRAINING XGBOOST FUSION CLASSIFIER ===")
    n_classes = len(classes)
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        eval_metric=eval_metric,
        early_stopping_rounds=20,
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    best_iter = model.best_iteration
    print(f"Best iteration: {best_iter}")
    return model


def evaluate(model, X, y, classes, split_name="Test"):
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average="macro", zero_division=0)
    auc   = roc_auc_score(y, proba, multi_class="ovr", average="macro") if len(classes) > 2 \
            else roc_auc_score(y, proba[:, 1])

    print(f"\n=== {split_name.upper()} EVALUATION ===")
    print(f"Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"F1 Macro : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, preds, target_names=classes, digits=4, zero_division=0))
    return preds, proba, acc, f1, auc


def run_shap(model, X, classes, feature_names, out_path):
    print("\n=== SHAP FEATURE IMPORTANCE ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:200])  # use up to 200 samples

    # For multi-class, shap_values is a list; pick the most-predicted class
    if isinstance(shap_values, list):
        sv = np.abs(shap_values[0])
        for s in shap_values[1:]:
            sv = np.maximum(sv, np.abs(s))
    else:
        sv = np.abs(shap_values)

    mean_shap = sv.mean(axis=0)
    top_idx   = mean_shap.argsort()[::-1][:15]

    print("Top 15 features by mean |SHAP|:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[i]:30s}  {mean_shap[i]:.5f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 6))
    labels  = [feature_names[i] for i in top_idx]
    vals    = [mean_shap[i] for i in top_idx]
    ax.barh(labels[::-1], vals[::-1], color="#e74c3c")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("MedFusion-Leuk — SHAP Feature Importance (Top 15)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"SHAP plot saved: {out_path}")
    return explainer


def run_baselines(X_tr, y_tr, X_te, y_te, classes):
    print("\n=== BASELINE COMPARISON ===")
    results = {}

    # SVM
    print("  Training SVM...")
    svm = SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
    svm.fit(X_tr, y_tr)
    p = svm.predict_proba(X_te)
    results["SVM"] = _score_row("SVM", y_te, p, classes)

    # Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)
    results["Random Forest"] = _score_row("Random Forest", y_te, p, classes)

    # CNN only (image dims only — first IMAGE_FEATURE_SIZE dims of X)
    print("  Training CNN-only baseline...")
    from config import IMAGE_FEATURE_SIZE
    X_tr_img = X_tr[:, :IMAGE_FEATURE_SIZE]
    X_te_img = X_te[:, :IMAGE_FEATURE_SIZE]
    rf_img = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_img.fit(X_tr_img, y_tr)
    p = rf_img.predict_proba(X_te_img)
    results["CNN only"] = _score_row("CNN only", y_te, p, classes)

    # NLP only (last NLP_FEATURE_SIZE dims)
    print("  Training NLP-only baseline...")
    from config import NLP_FEATURE_SIZE
    X_tr_nlp = X_tr[:, -NLP_FEATURE_SIZE:]
    X_te_nlp = X_te[:, -NLP_FEATURE_SIZE:]
    rf_nlp = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_nlp.fit(X_tr_nlp, y_tr)
    p = rf_nlp.predict_proba(X_te_nlp)
    results["BioBERT only"] = _score_row("BioBERT only", y_te, p, classes)

    return results


def _score_row(name, y_true, y_proba, classes):
    preds = y_proba.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average="macro", zero_division=0)
    rec  = recall_score(y_true, preds, average="macro", zero_division=0)
    f1   = f1_score(y_true, preds, average="macro", zero_division=0)
    auc  = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro") if len(classes) > 2 \
           else roc_auc_score(y_true, y_proba[:, 1])
    print(f"    {name:20s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Macro": f1, "AUC-ROC": auc}


def print_comparison_table(baselines: dict, medfusion_scores: dict):
    print("\n" + "="*75)
    print(f"{'Method':<22} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'AUC-ROC':>7}")
    print("-"*75)
    all_methods = {**baselines, "MedFusion-Leuk (ours)": medfusion_scores}
    for method, scores in all_methods.items():
        marker = " <--" if method == "MedFusion-Leuk (ours)" else ""
        print(f"  {method:<20} | {scores['Accuracy']:>8.3f} | {scores['Precision']:>9.3f} | "
              f"{scores['Recall']:>6.3f} | {scores['F1 Macro']:>6.3f} | {scores['AUC-ROC']:>7.3f}{marker}")
    print("="*75)


def plot_comparison(baselines: dict, medfusion_scores: dict, out_path):
    all_methods = {**baselines, "MedFusion-Leuk": medfusion_scores}
    metrics = ["Accuracy", "F1 Macro", "AUC-ROC"]
    colors  = ["#95a5a6"] * len(baselines) + ["#e74c3c"]
    df = pd.DataFrame(all_methods).T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        ax.barh(list(all_methods.keys()), df[metric].values, color=colors)
        ax.set_xlim(max(0, df[metric].min() - 0.05), 1.0)
        ax.set_title(metric, fontsize=12)
        ax.invert_yaxis()
        for i, v in enumerate(df[metric].values):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    plt.suptitle("Comparative Analysis: MedFusion-Leuk vs Baselines", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Comparison chart saved: {out_path}")


def main():
    X_tr, y_tr, X_va, y_va, X_te, y_te, classes = load_data()
    n_feats = X_tr.shape[1]
    feature_names = [f"fused_{i:03d}" for i in range(n_feats)]

    # Combine train+val for final training
    X_trainval = np.vstack([X_tr, X_va])
    y_trainval = np.concatenate([y_tr, y_va])

    xgb_model = train_xgboost(X_tr, y_tr, X_va, y_va, classes)

    preds, proba, acc, f1, auc = evaluate(xgb_model, X_te, y_te, classes)
    medfusion_scores = {
        "Accuracy":  acc,
        "Precision": precision_score(y_te, preds, average="macro", zero_division=0),
        "Recall":    recall_score(y_te, preds, average="macro", zero_division=0),
        "F1 Macro":  f1,
        "AUC-ROC":   auc,
    }

    explainer = run_shap(xgb_model, X_te, classes, feature_names, EVAL_DIR / "shap_summary.png")

    # Save model and explainer
    xgb_model.save_model(str(MODEL_DIR / "fusion_classifier.json"))
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")
    np.save(MODEL_DIR / "classes.npy", np.array(classes))
    print(f"\nModel saved: {MODEL_DIR / 'fusion_classifier.json'}")
    print(f"SHAP explainer saved: {MODEL_DIR / 'shap_explainer.pkl'}")

    baselines = run_baselines(X_tr, y_tr, X_te, y_te, classes)

    print_comparison_table(baselines, medfusion_scores)
    plot_comparison(baselines, medfusion_scores, EVAL_DIR / "comparative_analysis.png")

    # ROC curves
    from evaluation.roc_curve import plot_roc_curves
    fig = plot_roc_curves(y_te, proba, classes=classes, save_path=EVAL_DIR / "roc_curves.png")
    plt.close("all")
    print(f"ROC curves saved: {EVAL_DIR / 'roc_curves.png'}")

    print("\n=== PHASE 3 COMPLETE ===")


if __name__ == "__main__":
    main()
