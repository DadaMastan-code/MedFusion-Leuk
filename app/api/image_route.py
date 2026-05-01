"""FastAPI route: blood smear image upload and analysis."""

import io
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from config import LEUKEMIA_CLASSES

router = APIRouter(prefix="/image", tags=["Image Analysis"])

# Model loaded lazily on first request
_model = None
_gradcam = None


def _get_model():
    global _model, _gradcam
    if _model is None:
        import torch
        from modules.image_analysis.model import build_model
        from modules.image_analysis.gradcam import GradCAM
        _model = build_model(pretrained=True)
        _model.eval()
        _gradcam = GradCAM(_model, target_layer_name="efficientnet.blocks")
    return _model, _gradcam


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Classify a blood smear image and return prediction + Grad-CAM path."""
    if file.content_type not in {"image/jpeg", "image/png", "image/tiff"}:
        raise HTTPException(400, "Only JPG/PNG/TIFF images are accepted.")

    import torch
    from modules.image_analysis.preprocess import EVAL_TRANSFORMS
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    tensor = EVAL_TRANSFORMS(pil_img)

    model, gradcam = _get_model()
    with torch.no_grad():
        logits, features = model(tensor.unsqueeze(0))
        proba = torch.softmax(logits, dim=1).squeeze().tolist()

    class_idx = int(torch.tensor(proba).argmax())
    run_id = uuid.uuid4().hex[:8]
    cam_path = gradcam.save(pil_img, tensor, class_idx, filename=f"gradcam_{run_id}.png")

    return {
        "predicted_class": LEUKEMIA_CLASSES[class_idx],
        "confidence": proba[class_idx],
        "probabilities": dict(zip(LEUKEMIA_CLASSES, proba)),
        "gradcam_path": str(cam_path),
        "feature_vector_shape": list(features.shape),
    }
