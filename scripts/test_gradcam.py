"""
Step 4 — Test Grad-CAM visualization on 3 test images.
Loads the trained model, generates heatmaps, saves overlays.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(__file__[:__file__.rfind('/')].rstrip('/scripts') or '.'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib_ok = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_ok = False

from pathlib import Path
from PIL import Image

from modules.image_analysis.model import build_model
from modules.image_analysis.preprocess import EVAL_TRANSFORMS, load_image_pil
from modules.image_analysis.gradcam import GradCAM
from config import IMAGE_MODEL_DIR, GRADCAM_DIR

GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR = Path("data/processed/splits")

DEVICE = torch.device("cpu")  # GradCAM needs autograd; CPU is most reliable

print(f"Device: {DEVICE}")


def load_model():
    ckpt_path = IMAGE_MODEL_DIR / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run train_image_model.py first.")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes), pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded: val_acc={ckpt['val_acc']:.4f}  classes={classes}")
    return model, classes


def run_gradcam_test():
    model, classes = load_model()

    # GradCAM hooks into EfficientNet's last conv block
    gcam = GradCAM(model, target_layer_name="efficientnet.blocks")

    test_df = pd.read_csv(SPLIT_DIR / "test.csv")
    # Pick 1-2 samples per class
    samples = []
    for cls in classes:
        cls_rows = test_df[test_df.label == cls].head(2)
        samples.extend(cls_rows.to_dict("records"))
    samples = samples[:min(4, len(samples))]

    results = []
    for i, row in enumerate(samples):
        img_path = row["filepath"]
        true_label = row["label"]
        print(f"\n--- Sample {i+1}: {Path(img_path).name} ({true_label}) ---")

        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = EVAL_TRANSFORMS(pil_img).to(DEVICE)

        # Predict
        model.eval()
        with torch.no_grad():
            logits, feats = model(input_tensor.unsqueeze(0))
        proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_class = int(proba.argmax())
        pred_label = classes[pred_class]
        conf = proba[pred_class]
        print(f"  Predicted: {pred_label} ({conf:.1%})")
        print(f"  True:      {true_label}  {'✓ correct' if pred_label == true_label else '✗ wrong'}")
        print(f"  Proba:     {dict(zip(classes, proba.round(3)))}")
        print(f"  Features:  shape={feats.shape}  norm={feats.cpu().norm().item():.3f}")

        # Grad-CAM (uses gradient — need requires_grad)
        try:
            fname = f"gradcam_{i+1}_{true_label}.png"
            out_path = gcam.save(pil_img, input_tensor, class_idx=pred_class, filename=fname)
            print(f"  Grad-CAM saved: {out_path}")
            results.append({"sample": i+1, "true": true_label, "pred": pred_label,
                            "conf": conf, "gradcam": str(out_path)})
        except Exception as e:
            print(f"  Grad-CAM error: {e}")
            results.append({"sample": i+1, "true": true_label, "pred": pred_label,
                            "conf": conf, "gradcam": None})

    print("\n=== GRAD-CAM TEST COMPLETE ===")
    correct = sum(1 for r in results if r["true"] == r["pred"])
    print(f"Accuracy on {len(results)} samples: {correct}/{len(results)} ({correct/len(results):.0%})")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    run_gradcam_test()
