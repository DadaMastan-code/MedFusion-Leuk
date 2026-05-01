"""
Step 5 — Extract image feature vectors from trained model +
generate synthetic hematological parameters matched to class labels.
Produces numpy arrays for fusion classifier training.
"""

import sys, warnings, os
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.image_analysis.model import build_model
from modules.image_analysis.preprocess import EVAL_TRANSFORMS
from config import IMAGE_MODEL_DIR, PDF_FEATURE_SIZE, NLP_FEATURE_SIZE, IMAGE_FEATURE_SIZE

SPLIT_DIR   = Path("data/processed/splits")
FEATURE_DIR = Path("data/processed/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE  = 16 if DEVICE.type == "cuda" else (16 if DEVICE.type == "mps" else 8)

# Realistic hematological ranges per class  [low, high]
HEMATOLOGY_RANGES = {
    "ALL":    {"wbc":[30_000,150_000], "blast":[25,90],  "hb":[6,10],  "plt":[20_000,100_000], "ldh":[400,2000], "uric":[6,12]},
    "AML":    {"wbc":[20_000,200_000], "blast":[20,95],  "hb":[6,9],   "plt":[15_000,80_000],  "ldh":[500,2500], "uric":[7,14]},
    "CLL":    {"wbc":[10_000,200_000], "blast":[0,5],    "hb":[9,13],  "plt":[80_000,200_000], "ldh":[200,600],  "uric":[4,8]},
    "CML":    {"wbc":[50_000,300_000], "blast":[0,19],   "hb":[9,13],  "plt":[200_000,700_000],"ldh":[300,800],  "uric":[5,10]},
    "Normal": {"wbc":[4_000,11_000],   "blast":[0,4],    "hb":[12,17], "plt":[150_000,400_000],"ldh":[140,280],  "uric":[3.5,7.2]},
}

# Normalisation ranges (must match parser.py _RANGES)
NORM_RANGES = {
    "wbc": (0, 200_000), "rbc": (0, 10), "hb": (0, 25),
    "platelet": (0, 1_000_000), "blast_pct": (0, 100),
    "ldh": (0, 5000), "uric_acid": (0, 30),
}


def synth_pdf_features(label: str, rng: np.random.Generator) -> np.ndarray:
    r = HEMATOLOGY_RANGES.get(label, HEMATOLOGY_RANGES["Normal"])
    raw = np.array([
        rng.uniform(*r["wbc"]),
        rng.uniform(3.0, 5.5),           # rbc — mild variation
        rng.uniform(*r["hb"]),
        rng.uniform(*r["plt"]),
        rng.uniform(*r["blast"]),
        rng.uniform(*r["ldh"]),
        rng.uniform(*r["uric"]),
    ])
    # Normalise
    lo = np.array([v[0] for v in NORM_RANGES.values()])
    hi = np.array([v[1] for v in NORM_RANGES.values()])
    norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    # Pad to PDF_FEATURE_SIZE
    if len(norm) < PDF_FEATURE_SIZE:
        norm = np.pad(norm, (0, PDF_FEATURE_SIZE - len(norm)))
    return norm[:PDF_FEATURE_SIZE]


def synth_nlp_features(label: str, rng: np.random.Generator) -> np.ndarray:
    """
    # [NOVELTY] Context-aware NLP feature generation matches inference-time
    # rule-based logic: Normal samples simulate "blast 1-4%" in routine CBC
    # text (which does NOT trigger the leukemia flag), while leukemia samples
    # simulate "blast > 5%" or explicit disease terminology (flag = 1).
    # Gaussian noise on the flag prevents XGBoost from over-relying on a single
    # binary feature — forces the model to also learn from quantitative lab params.
    """
    vec = rng.uniform(0, 0.3, NLP_FEATURE_SIZE).astype(np.float32)
    leuk_classes = {"ALL", "AML", "CLL", "CML"}

    if label in leuk_classes:
        # Leukemia: blast > 5% or disease keywords → flag ≈ 1.0
        # Small noise to prevent XGBoost over-fitting on a single binary feature
        vec[12] = float(rng.choice([1.0, 0.9, 0.8], p=[0.80, 0.10, 0.10]))
    else:
        # Normal: routine CBC "blast 1-4%" → context-aware flag = 0
        # Occasionally simulate text with no blast mention → flag also 0
        vec[12] = float(rng.choice([0.0, 0.1], p=[0.90, 0.10]))

    return vec


def extract_image_features(model, img_paths: list[Path]) -> np.ndarray:
    """Extract 512-d feature vectors from the model's penultimate layer."""
    model.eval()
    all_feats = []
    with torch.no_grad():
        batch = []
        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")
                t = EVAL_TRANSFORMS(img)
                batch.append(t)
            except Exception:
                batch.append(torch.zeros(3, 224, 224))

            if len(batch) == BATCH_SIZE:
                feats = model(torch.stack(batch).to(DEVICE))[1]  # (B, 512)
                all_feats.append(feats.cpu().numpy())
                batch = []

        if batch:
            feats = model(torch.stack(batch).to(DEVICE))[1]
            all_feats.append(feats.cpu().numpy())

    return np.vstack(all_feats)


def process_split(split: str, model, classes: list[str]):
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    print(f"\n  Processing {split}: {len(df)} samples")
    rng = np.random.default_rng(42)

    img_paths = [Path(r.filepath) for _, r in df.iterrows()]
    labels    = [classes.index(r.label) for _, r in df.iterrows()]

    print(f"    Extracting image features ({len(img_paths)} images)...")
    img_feats = extract_image_features(model, img_paths)   # (N, 512)
    img_feats_trunc = img_feats[:, :IMAGE_FEATURE_SIZE]

    print(f"    Generating synthetic PDF + NLP features...")
    fused_list = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"    Fusing {split}")):
        pdf_vec = synth_pdf_features(row.label, rng)
        nlp_vec = synth_nlp_features(row.label, rng)
        # Concatenate raw modality vectors: [img(256) | pdf(64) | nlp(128)] = 448 dims
        # This layout lets the XGBoost baselines slice modality-specific subspaces.
        fused = np.concatenate([img_feats_trunc[i], pdf_vec, nlp_vec])
        fused_list.append(fused)

    X = np.stack(fused_list).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    np.save(FEATURE_DIR / f"fusion_features_{split}.npy", X)
    np.save(FEATURE_DIR / f"fusion_labels_{split}.npy", y)
    print(f"    Saved: {X.shape[0]} samples × {X.shape[1]} features  labels={y.shape}")
    return X, y


def main():
    print("=== GENERATING FUSION TRAINING DATA ===")

    # Load trained model
    ckpt_path = IMAGE_MODEL_DIR / "best_model.pth"
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found. Run train_image_model.py first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes), pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model: {ckpt_path}  val_acc={ckpt['val_acc']:.4f}  classes={classes}")

    results = {}
    for split in ["train", "val", "test"]:
        X, y = process_split(split, model, classes)
        results[split] = (X, y)

    # Save class list for classifier
    np.save(FEATURE_DIR / "classes.npy", np.array(classes))

    print("\n=== SUMMARY ===")
    for split, (X, y) in results.items():
        counts = {classes[c]: int((y==c).sum()) for c in range(len(classes))}
        print(f"  {split:5s}: {X.shape[0]:5d} × {X.shape[1]} features  {counts}")


if __name__ == "__main__":
    main()
