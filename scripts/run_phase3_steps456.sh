#!/bin/bash
# Phase 3 Steps 4, 5, 6: GradCAM → Fusion Data → XGBoost
# Run after train_image_model.py completes.
set -e
cd "$(dirname "$0")/.."
echo "=== STEP 4: GRAD-CAM VISUALIZATION ==="
python -u scripts/test_gradcam.py

echo ""
echo "=== STEP 5: GENERATE FUSION TRAINING DATA ==="
python -u scripts/generate_fusion_training_data.py

echo ""
echo "=== STEP 6: TRAIN XGBOOST FUSION CLASSIFIER ==="
python -u scripts/train_fusion_classifier.py

echo ""
echo "=== PHASE 3 COMPLETE ==="
