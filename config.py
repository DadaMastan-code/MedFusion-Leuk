"""
MedFusion-Leuk — Central configuration constants.
All thresholds, paths, and model IDs live here.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STATIC_DIR = ROOT_DIR / "static"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LITERATURE_DIR = DATA_DIR / "literature"

IMAGE_MODEL_DIR = MODELS_DIR / "efficientnet_vit"
XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost"
BIOBERT_MODEL_DIR = MODELS_DIR / "biobert"

SHAP_PLOT_DIR = STATIC_DIR / "shap_plots"
GRADCAM_DIR = STATIC_DIR / "gradcam"

FAISS_INDEX_PATH = DATA_DIR / "literature" / "faiss_index"

# ── Model IDs ──────────────────────────────────────────────────────────────
EFFICIENTNET_VARIANT = "efficientnet_b0"          # timm identifier (B0=4M params, memory-efficient)
VIT_VARIANT = "vit_tiny_patch16_224"              # timm identifier (Tiny=5.5M params)
BIOBERT_MODEL_ID = "d4data/biomedical-ner-all"   # fine-tuned biomedical NER
SENTENCE_TRANSFORMER_ID = "all-MiniLM-L6-v2"

# ── LLM configuration (local, free — no API key required) ──────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "meditron")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Image settings ─────────────────────────────────────────────────────────
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_FEATURE_DIM = 1280   # EfficientNet-B0 penultimate layer

# ── Leukemia classes ───────────────────────────────────────────────────────
LEUKEMIA_CLASSES = ["Normal", "ALL", "AML", "CLL", "CML"]
NUM_CLASSES = len(LEUKEMIA_CLASSES)

# ── Clinical risk thresholds ───────────────────────────────────────────────
RISK_THRESHOLDS = {
    "wbc_high": 50_000,        # /µL  → High risk weight
    "blast_very_high": 60.0,   # %    → Very High risk weight
    "blast_high": 20.0,        # %    → High risk weight
    "hb_low": 9.0,             # g/dL → Medium risk weight
    "platelet_low": 80_000,    # /µL  → Medium risk weight
    "ldh_high": 600,           # U/L  → Medium risk weight
    "wbc_moderate": 20_000,    # /µL  → Moderate risk weight
}

RISK_WEIGHTS = {
    "wbc_high": 3,
    "blast_very_high": 4,
    "blast_high": 2,
    "hb_low": 2,
    "platelet_low": 2,
    "ldh_high": 1,
    "wbc_moderate": 1,
}

RISK_SCORE_BANDS = {
    "Very High": 6,
    "High": 4,
    "Moderate": 2,
    "Low": 0,
}

# ── ICD-10 codes ───────────────────────────────────────────────────────────
ICD10_MAP = {
    "ALL": "C91.0",
    "AML": "C92.0",
    "CLL": "C91.1",
    "CML": "C92.1",
    "Normal": "Z03.89",
}

ICD10_DESCRIPTIONS = {
    "C91.0": "Acute lymphoblastic leukemia [ALL]",
    "C92.0": "Acute myeloid leukemia [AML]",
    "C91.1": "Chronic lymphocytic leukemia of B-cell type [CLL]",
    "C92.1": "Chronic myeloid leukemia, BCR/ABL-positive [CML]",
    "Z03.89": "Encounter for observation for other suspected diseases and conditions ruled out",
}

# ── LOINC codes ────────────────────────────────────────────────────────────
LOINC_MAP = {
    "wbc": {"code": "6690-2", "display": "Leukocytes [#/volume] in Blood"},
    "rbc": {"code": "789-8",  "display": "Erythrocytes [#/volume] in Blood"},
    "hb":  {"code": "718-7",  "display": "Hemoglobin [Mass/volume] in Blood"},
    "platelet": {"code": "777-3",  "display": "Platelets [#/volume] in Blood"},
    "blast":    {"code": "26444-0","display": "Basophils [#/volume] in Blood (used for blast %)"},
    "ldh":      {"code": "2532-0", "display": "Lactate dehydrogenase [Enzymatic activity/volume] in Serum"},
    "uric_acid":{"code": "3084-1", "display": "Urate [Mass/volume] in Serum or Plasma"},
}

# ── RAG settings ───────────────────────────────────────────────────────────
RAG_TOP_K = 3
RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 64

# ── XGBoost ────────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
CV_FOLDS = 5

# ── Fusion ─────────────────────────────────────────────────────────────────
IMAGE_FEATURE_SIZE = 256    # projected dimension after fusion adapter
PDF_FEATURE_SIZE = 64
NLP_FEATURE_SIZE = 128
FUSED_FEATURE_SIZE = IMAGE_FEATURE_SIZE + PDF_FEATURE_SIZE + NLP_FEATURE_SIZE

# ── API ────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 50

# ── Chatbot ────────────────────────────────────────────────────────────────
MAX_CONVERSATION_TURNS = 20

SYSTEM_PROMPT = """You are MedFusion-Leuk, a clinical AI assistant specializing in
leukemia detection and hematological analysis. You support physicians and researchers
by synthesizing multimodal diagnostic data — blood smear images, CBC reports, and
clinical notes — into clear, evidence-based clinical summaries.

You always:
- Provide diagnosis with confidence score and risk level
- Cite relevant medical literature
- Include ICD-10 and LOINC codes in structured output
- Explain SHAP feature contributions in plain English
- Add the disclaimer: "This is a clinical decision SUPPORT tool. Final diagnosis
  must be confirmed by a licensed hematologist."

You never fabricate lab values, invent citations, or provide treatment prescriptions."""

# ── Abbreviation expansion ─────────────────────────────────────────────────
ABBREVIATION_MAP = {
    "WBC": "White Blood Cell",
    "RBC": "Red Blood Cell",
    "Hb": "Hemoglobin",
    "Hgb": "Hemoglobin",
    "plt": "Platelet",
    "PLT": "Platelet",
    "LDH": "Lactate Dehydrogenase",
    "ALL": "Acute Lymphoblastic Leukemia",
    "AML": "Acute Myeloid Leukemia",
    "CLL": "Chronic Lymphocytic Leukemia",
    "CML": "Chronic Myeloid Leukemia",
    "CBC": "Complete Blood Count",
    "BM": "Bone Marrow",
    "PB": "Peripheral Blood",
    "MCV": "Mean Corpuscular Volume",
    "MCH": "Mean Corpuscular Hemoglobin",
    "MCHC": "Mean Corpuscular Hemoglobin Concentration",
    "ESR": "Erythrocyte Sedimentation Rate",
    "PT": "Prothrombin Time",
    "aPTT": "Activated Partial Thromboplastin Time",
    "INR": "International Normalized Ratio",
    "ANC": "Absolute Neutrophil Count",
}
