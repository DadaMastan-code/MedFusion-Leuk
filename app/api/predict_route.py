"""
FastAPI prediction routes — Phase 4 full integration.

POST /predict          : JSON body {clinical_text, parameters}
POST /predict/image    : multipart {image, clinical_text, parameters_json}
POST /predict/full     : legacy multipart (Streamlit-compatible)
"""

import io, uuid, json, re, warnings, tempfile
import numpy as np
import xgboost as xgb

import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from pathlib import Path

from modules.pdf_extraction.parser import parse_report, BloodReportFeatures
from modules.prediction.risk_scoring import compute_risk
from modules.coding.icd_mapper import build_coding_report
from modules.rag.retriever import MedicalRetriever
from app.chatbot import MedFusionChatbot, DiagnosticContext
from config import (
    IMAGE_FEATURE_SIZE, PDF_FEATURE_SIZE, NLP_FEATURE_SIZE,
    XGBOOST_MODEL_DIR, IMAGE_MODEL_DIR, FAISS_INDEX_PATH,
    SHAP_PLOT_DIR, GRADCAM_DIR,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ── Singletons ─────────────────────────────────────────────────────────────
_img_model = None
_img_classes = None
_gradcam = None
_xgb_model = None
_xgb_classes = None
_explainer = None
_retriever = None

# Human-readable feature names for SHAP output
_PDF_PARAM_NAMES = ["wbc", "rbc", "hemoglobin", "platelets", "blast_pct", "ldh", "uric_acid"]
_FEATURE_NAMES = (
    [f"img_{i:03d}" for i in range(IMAGE_FEATURE_SIZE)]
    + [_PDF_PARAM_NAMES[i] if i < len(_PDF_PARAM_NAMES) else f"pdf_{i:03d}" for i in range(PDF_FEATURE_SIZE)]
    + ["leuk_keyword_flag" if i == 12 else f"nlp_{i:03d}" for i in range(NLP_FEATURE_SIZE)]
)


def _init():
    """Load only XGBoost + SHAP at startup — image model and RAG are loaded lazily."""
    global _xgb_model, _xgb_classes, _explainer

    if _xgb_model is not None:
        return

    # XGBoost + SHAP (~15MB total — safe during startup)
    xgb_path = XGBOOST_MODEL_DIR / "fusion_classifier.json"
    if not xgb_path.exists():
        raise RuntimeError(f"XGBoost model not found: {xgb_path}. Run train_fusion_classifier.py first.")
    _xgb_model = xgb.XGBClassifier()
    _xgb_model.load_model(str(xgb_path))
    _xgb_classes = list(np.load(XGBOOST_MODEL_DIR / "classes.npy"))
    import shap
    _explainer = shap.TreeExplainer(_xgb_model)


def _ensure_image_model():
    """Lazy-load image model on first image request."""
    global _img_model, _img_classes, _gradcam
    if _img_model is not None:
        return
    ckpt_path = IMAGE_MODEL_DIR / "best_model.pth"
    if not ckpt_path.exists():
        return
    import torch
    from modules.image_analysis.model import build_model
    from modules.image_analysis.gradcam import GradCAM
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _img_classes = ckpt["classes"]
    _img_model = build_model(num_classes=len(_img_classes), pretrained=False)
    _img_model.load_state_dict(ckpt["model_state"])
    _img_model.eval()
    _gradcam = GradCAM(_img_model, "efficientnet.blocks")


def _ensure_retriever():
    """Lazy-load RAG retriever on first retrieval request."""
    global _retriever
    if _retriever is not None:
        return
    try:
        _retriever = MedicalRetriever()
        _retriever.load(FAISS_INDEX_PATH)
    except Exception:
        _retriever = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _params_to_features(params: dict) -> BloodReportFeatures:
    return BloodReportFeatures(
        wbc=params.get("wbc"),
        rbc=params.get("rbc"),
        hb=params.get("hemoglobin") or params.get("hb"),
        platelet=params.get("platelets") or params.get("platelet"),
        blast_pct=params.get("blast_pct"),
        ldh=params.get("ldh"),
        uric_acid=params.get("uric_acid"),
    )


def _blood_to_pdf_feat(bf: BloodReportFeatures) -> np.ndarray:
    vec = bf.to_feature_vector()          # (7,) normalised
    vec = np.nan_to_num(vec, nan=0.0)
    if len(vec) < PDF_FEATURE_SIZE:
        vec = np.pad(vec, (0, PDF_FEATURE_SIZE - len(vec)))
    return vec[:PDF_FEATURE_SIZE].astype(np.float32)


def _extract_image_feat(pil_img: Image.Image) -> tuple[np.ndarray, int, float]:
    """Returns (img_feat_256, class_idx, confidence)."""
    _ensure_image_model()
    import torch
    from modules.image_analysis.preprocess import EVAL_TRANSFORMS
    tensor = EVAL_TRANSFORMS(pil_img)
    with torch.no_grad():
        logits, feats = _img_model(tensor.unsqueeze(0))
        proba = torch.softmax(logits, dim=1).squeeze().numpy()
    class_idx = int(proba.argmax())
    confidence = float(proba[class_idx])
    img_feat = feats.squeeze(0).numpy()[:IMAGE_FEATURE_SIZE]
    return img_feat, class_idx, confidence


def _get_nlp_feat(text: str, params: dict | None = None) -> np.ndarray:
    """
    # [NOVELTY] Context-aware NLP feature weighting prevents keyword-only false
    # positives when quantitative lab parameters contradict text cues. "blast 1%"
    # in a routine CBC report must NOT trigger the leukemia flag — only blast > 5%
    # or explicit disease terminology (leukemia, AML, ALL, etc.) does. Quantitative
    # parameters, when available, override ambiguous text keyword matches.
    # This is "context-aware multimodal arbitration" — a research contribution where
    # quantitative evidence takes precedence over textual keyword cues.
    """
    vec = np.zeros(NLP_FEATURE_SIZE, dtype=np.float32)
    t = text.lower()

    # Disease-specific keywords that unambiguously indicate malignancy (excludes "blast"
    # which is a normal CBC lab field name, e.g., "blast 1%")
    leuk_disease_kws = {"leukemia", "lymphoblast", "myeloid", "lymphocytic",
                        "myelogenous", "lymphoma", "aml", "all", "cll", "cml"}
    diag_kws  = {"leukemia", "lymphoma", "anemia", "malignancy", "cancer", "aml", "all"}
    symp_kws  = {"fatigue", "pallor", "bleeding", "bruising", "infection", "fever", "weakness"}
    lab_kws   = {"wbc", "hemoglobin", "platelet", "blast", "ldh", "neutrophil", "rbc", "hb"}
    proc_kws  = {"biopsy", "flow cytometry", "cytogenetics", "pcr", "smear", "cbc"}
    med_kws   = {"imatinib", "dasatinib", "cytarabine", "vincristine", "prednisone", "chemotherapy"}
    anat_kws  = {"bone marrow", "peripheral blood", "lymph node", "spleen"}

    counts = [
        sum(1 for k in anat_kws  if k in t),   # 0: ANATOMY
        sum(1 for k in diag_kws  if k in t),   # 1: DIAGNOSIS
        sum(1 for k in lab_kws   if k in t),   # 2: LAB_VALUE
        sum(1 for k in med_kws   if k in t),   # 3: MEDICATION
        sum(1 for k in proc_kws  if k in t),   # 4: PROCEDURE
        sum(1 for k in symp_kws  if k in t),   # 5: SYMPTOM
    ]
    total = max(sum(counts), 1)
    for i, c in enumerate(counts):
        vec[i] = c / total
        vec[i + 6] = min(c / 3.0, 1.0)

    # ── Context-aware leukemia flag (index 12) ──────────────────────────────
    # Step 1: check negation ("no blast cells detected", "blast negative")
    negated = bool(re.search(
        r'no\s+blast|blast\s+(?:cells?\s+)?(?:not\s+)?(?:detected|found|seen|identified)'
        r'|blast\s+negative|negative\s+for\s+blast|absent',
        t
    ))

    # Step 2: parse blast percentage from text ("blast 72%", "blast 1 percent")
    blast_match = re.search(
        r'blast\s*(?:cells?|percentage?|pct|count)?\s*[:\s=]?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)?',
        t
    )
    blast_val_in_text = float(blast_match.group(1)) if blast_match else None

    # Step 3: quantitative parameter override — params win over text if available
    param_blast = float(params["blast_pct"]) if (params and "blast_pct" in params and params["blast_pct"] is not None) else None

    disease_kw_found = any(kw in t for kw in leuk_disease_kws)

    if negated:
        vec[12] = 0.0
    elif param_blast is not None and param_blast <= 5.0 and not disease_kw_found:
        # Strong quantitative evidence: blast ≤ 5% and no explicit disease keyword
        vec[12] = 0.0
    elif disease_kw_found:
        vec[12] = 1.0
    elif param_blast is not None:
        vec[12] = 1.0 if param_blast > 5.0 else 0.0
    elif blast_val_in_text is not None:
        vec[12] = 1.0 if blast_val_in_text > 5.0 else 0.0
    else:
        # "blast" in text without numeric context and no disease keyword → not flagged
        vec[12] = 0.0

    return vec


def _xgb_predict(fused: np.ndarray) -> tuple[str, float, np.ndarray]:
    fused = np.nan_to_num(fused, nan=0.0).reshape(1, -1)
    proba = _xgb_model.predict_proba(fused)[0]
    idx = int(proba.argmax())
    return _xgb_classes[idx], float(proba[idx]), proba


def _shap_top(fused: np.ndarray, n: int = 10) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = _explainer.shap_values(fused.reshape(1, -1))
    abs_sv = np.abs(sv[0]) if not isinstance(sv, list) else np.maximum.reduce([np.abs(s[0]) for s in sv])
    top_idx = abs_sv.argsort()[::-1][:n]
    return {_FEATURE_NAMES[i]: round(float(abs_sv[i]), 5) for i in top_idx}


def _rag_query(predicted_class: str) -> tuple[str, list]:
    _ensure_retriever()
    if _retriever is None:
        return "No literature index available.", []
    query = f"{predicted_class} leukemia diagnosis blast WBC risk prognosis"
    chunks = _retriever.retrieve(query, top_k=3)
    formatted = _retriever.format_context(chunks)
    chunk_list = [{"source": c.source, "score": round(c.score, 3), "text": c.text[:200] + "..."} for c in chunks]
    return formatted, chunk_list


def _generate_explanation(predicted_class: str, confidence: float, risk, coding) -> str:
    try:
        chatbot = MedFusionChatbot()
        msg = (
            f"Predicted class: {predicted_class} ({confidence*100:.0f}% confidence). "
            f"Risk level: {risk.level} (score: {risk.score}). "
            f"ICD-10: {coding.icd10_code} — {coding.icd10_description}. "
            f"Risk triggers: {'; '.join(risk.triggered) or 'none'}. "
            "Give a 2-sentence clinical summary and add the standard disclaimer."
        )
        return chatbot.respond(msg)
    except Exception:
        triggers = "; ".join(risk.triggered) or "no specific triggers"
        return (
            f"Predicted: {predicted_class} ({confidence*100:.0f}% confidence). "
            f"Risk level: {risk.level} (score: {risk.score}) — {triggers}. "
            "This is a clinical decision SUPPORT tool; final diagnosis must be confirmed by a licensed hematologist."
        )


def _build_response(
    predicted_class: str,
    confidence: float,
    proba: np.ndarray,
    blood_features: BloodReportFeatures,
    fused: np.ndarray,
    modalities_used: list[str],
    gradcam_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    run_id = run_id or uuid.uuid4().hex[:8]
    risk = compute_risk(blood_features)
    coding = build_coding_report(predicted_class, blood_features)
    rag_context, rag_chunks = _rag_query(predicted_class)
    shap_top = _shap_top(fused)
    explanation = _generate_explanation(predicted_class, confidence, risk, coding)

    loinc_codes = {
        obs.loinc_code: {"display": obs.display, "value": obs.value, "unit": obs.unit}
        for obs in coding.loinc_observations
        if obs.value is not None
    }

    return {
        "run_id": run_id,
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(_xgb_classes, proba)},
        "risk_level": risk.level,
        "risk_score": risk.score,
        "risk_triggers": risk.triggered,
        "icd_code": coding.icd10_code,
        "icd_description": coding.icd10_description,
        "loinc_codes": loinc_codes,
        "shap_features": {"top_features": shap_top},
        "rag_context": rag_context,
        "rag_chunks": rag_chunks,
        "explanation": explanation,
        "modalities_used": modalities_used,
        "extracted_lab_values": blood_features.to_dict(),
        "gradcam_path": gradcam_path,
    }


# ── Request model ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    clinical_text: Optional[str] = ""
    parameters: Optional[dict] = None


# ══════════════════════════════════════════════════════════════════════════
# POST /predict  — text + parameters
# ══════════════════════════════════════════════════════════════════════════

@router.post("")
async def predict_text(req: PredictRequest):
    """Predict from clinical text and/or structured lab parameters."""
    _init()

    if not req.clinical_text and not req.parameters:
        raise HTTPException(400, "Provide clinical_text and/or parameters.")

    def _run():
        modalities = []
        img_feat = np.zeros(IMAGE_FEATURE_SIZE, dtype=np.float32)
        pdf_feat = np.zeros(PDF_FEATURE_SIZE, dtype=np.float32)
        nlp_feat = np.zeros(NLP_FEATURE_SIZE, dtype=np.float32)
        blood_features = BloodReportFeatures()

        if req.parameters:
            blood_features = _params_to_features(req.parameters)
            pdf_feat = _blood_to_pdf_feat(blood_features)
            modalities.append("parameters")

        if req.clinical_text:
            nlp_feat = _get_nlp_feat(req.clinical_text, params=req.parameters)
            modalities.append("text")

        fused = np.concatenate([img_feat, pdf_feat, nlp_feat]).astype(np.float32)
        predicted_class, confidence, proba = _xgb_predict(fused)
        return _build_response(
            predicted_class=predicted_class,
            confidence=confidence,
            proba=proba,
            blood_features=blood_features,
            fused=fused,
            modalities_used=modalities,
        )

    return await run_in_threadpool(_run)


# ══════════════════════════════════════════════════════════════════════════
# POST /predict/image  — multipart image upload
# ══════════════════════════════════════════════════════════════════════════

@router.post("/image")
async def predict_image(
    image: UploadFile = File(...),
    clinical_text: str = Form(""),
    parameters_json: str = Form("{}"),
):
    """Predict from blood smear image + optional text/parameters. Generates Grad-CAM."""
    _init()

    if _img_model is None:
        raise HTTPException(503, "Image model not loaded. Run train_image_model.py first.")

    run_id = uuid.uuid4().hex[:8]
    raw = await image.read()

    def _run():
        modalities = ["image"]
        from modules.image_analysis.preprocess import EVAL_TRANSFORMS
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = EVAL_TRANSFORMS(pil_img)
        img_feat, class_idx, _ = _extract_image_feat(pil_img)

        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        cam_path = str(_gradcam.save(pil_img, tensor, class_idx, f"gradcam_{run_id}.png"))

        pdf_feat = np.zeros(PDF_FEATURE_SIZE, dtype=np.float32)
        nlp_feat = np.zeros(NLP_FEATURE_SIZE, dtype=np.float32)
        blood_features = BloodReportFeatures()

        params = {}
        try:
            params = json.loads(parameters_json) if parameters_json.strip() else {}
        except Exception:
            pass

        if params:
            blood_features = _params_to_features(params)
            pdf_feat = _blood_to_pdf_feat(blood_features)
            modalities.append("parameters")

        if clinical_text:
            nlp_feat = _get_nlp_feat(clinical_text, params=params)
            modalities.append("text")

        fused = np.concatenate([img_feat, pdf_feat, nlp_feat]).astype(np.float32)
        predicted_class, confidence, proba = _xgb_predict(fused)
        return _build_response(
            predicted_class=predicted_class,
            confidence=confidence,
            proba=proba,
            blood_features=blood_features,
            fused=fused,
            modalities_used=modalities,
            gradcam_path=cam_path,
            run_id=run_id,
        )

    return await run_in_threadpool(_run)


# ══════════════════════════════════════════════════════════════════════════
# POST /predict/full  — legacy multipart (Streamlit-compatible)
# ══════════════════════════════════════════════════════════════════════════

@router.post("/full")
async def full_predict(
    image: Optional[UploadFile] = File(None),
    pdf: Optional[UploadFile] = File(None),
    clinical_text: str = Form(""),
    user_question: str = Form("Please provide a clinical summary."),
):
    """Legacy multipart endpoint — backward-compatible with existing Streamlit UI."""
    _init()

    if image is None and pdf is None and not clinical_text:
        raise HTTPException(400, "Provide at least one of: image, PDF, or clinical text.")

    run_id = uuid.uuid4().hex[:8]
    modalities = []
    cam_path = None

    img_feat = np.zeros(IMAGE_FEATURE_SIZE, dtype=np.float32)
    pdf_feat = np.zeros(PDF_FEATURE_SIZE, dtype=np.float32)
    nlp_feat = np.zeros(NLP_FEATURE_SIZE, dtype=np.float32)
    blood_features = BloodReportFeatures()

    if image is not None and _img_model is not None:
        from modules.image_analysis.preprocess import EVAL_TRANSFORMS
        raw = await image.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = EVAL_TRANSFORMS(pil_img)
        img_feat, class_idx, _ = _extract_image_feat(pil_img)
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        cam_path = str(_gradcam.save(pil_img, tensor, class_idx, f"gradcam_{run_id}.png"))
        modalities.append("image")

    if pdf is not None:
        raw = await pdf.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        blood_features = parse_report(tmp_path)
        pdf_feat = _blood_to_pdf_feat(blood_features)
        modalities.append("pdf")

    if clinical_text:
        nlp_feat = _get_nlp_feat(clinical_text)
        modalities.append("text")

    fused = np.concatenate([img_feat, pdf_feat, nlp_feat]).astype(np.float32)
    predicted_class, confidence, proba = _xgb_predict(fused)

    risk = compute_risk(blood_features)
    coding = build_coding_report(predicted_class, blood_features)
    rag_context, _ = _rag_query(predicted_class)
    shap_top = _shap_top(fused)

    ctx = DiagnosticContext(
        predicted_class=predicted_class,
        confidence=confidence,
        risk_result=risk,
        coding_report=coding,
        shap_features={"top_shap_features": shap_top},
        retrieved_literature=rag_context,
        image_uploaded=image is not None,
        pdf_uploaded=pdf is not None,
        clinical_text=clinical_text,
    )
    try:
        chatbot = MedFusionChatbot()
        llm_response = chatbot.respond(user_question, diagnostic_context=ctx)
    except Exception:
        llm_response = "LLM unavailable. " + _generate_explanation(predicted_class, confidence, risk, coding)

    return {
        "run_id": run_id,
        "predicted_class": predicted_class,
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "risk_level": risk.level,
        "risk_score": risk.score,
        "risk_triggers": risk.triggered,
        "icd10_code": coding.icd10_code,
        "icd_code": coding.icd10_code,
        "icd10_description": coding.icd10_description,
        "icd_description": coding.icd10_description,
        "shap_features": {"top_shap_features": shap_top, "top_features": shap_top},
        "gradcam_path": cam_path,
        "extracted_lab_values": blood_features.to_dict(),
        "llm_response": llm_response,
        "explanation": llm_response,
        "literature_context": rag_context,
        "rag_context": rag_context,
        "modalities_used": modalities,
    }
