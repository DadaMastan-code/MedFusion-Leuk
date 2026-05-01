"""FastAPI route: blood report PDF upload and parameter extraction."""

import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException

from modules.pdf_extraction.parser import parse_report
from modules.prediction.risk_scoring import compute_risk

router = APIRouter(prefix="/pdf", tags=["PDF Extraction"])


@router.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract hematological parameters from an uploaded blood report PDF."""
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are accepted.")

    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    features = parse_report(tmp_path)
    risk = compute_risk(features)

    return {
        "extracted_values": features.to_dict(),
        "risk_score": risk.score,
        "risk_level": risk.level,
        "risk_triggers": risk.triggered,
        "feature_vector": features.to_feature_vector().tolist(),
    }
