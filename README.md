# MedFusion-Leuk

**Multimodal Fusion Architecture for Leukemia Clinical Decision Support**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-22%2F22-brightgreen)](#testing)

MedFusion-Leuk is a research-grade clinical decision support system for leukemia detection. It fuses five modalities — blood smear images, lab parameters, clinical text, medical literature (RAG), and LLM explanations — into a single end-to-end pipeline with full explainability (Grad-CAM + SHAP).

---

## Results

| Model | Accuracy | F1 Macro | AUC-ROC |
|---|---|---|---|
| CNN only (EfficientNet-B0) | 95.00% | 94.16% | 98.29% |
| SVM (full features) | 99.88% | 99.86% | 100.00% |
| Random Forest (full features) | 100.00% | 100.00% | 100.00% |
| NLP only (keyword, 128-dim) | 100.00% | 100.00% | 100.00% |
| **MedFusion-Leuk (all modalities)** | **100.00%** | **100.00%** | **100.00%** |

**Image model:** EfficientNet-B0 + ViT-Tiny hybrid — **94.69% test accuracy**, 95.62% best val (epoch 15)  
**Fusion classifier:** XGBoost on 448-dim fused vector — **100% accuracy**, **AUC 1.000**  
**Dataset:** C-NMC 2019, 10,661 images, ALL vs Normal

---

## Architecture

```
Blood Smear Image ──► EfficientNet-B0 + ViT-Tiny ──► img_feat (256-dim)
                                                            │
Lab Parameters ──────► Clinical Parser ──────────────► pdf_feat  (64-dim)  ──► XGBoost ──► Prediction
                                                            │                  (448-dim)     + Risk Level
Clinical Text ───────► Context-aware NLP ────────────► nlp_feat (128-dim)               + ICD-10 / LOINC
                                                            │
Medical Literature ──► FAISS RAG ────────────────────► Literature context
                                                            │
                        Ollama / meditron ───────────► Clinical explanation
                                                            │
                        SHAP + Grad-CAM ─────────────► Explainability
```

**Fused feature vector:** `[img(256) | pdf(64) | nlp(128)] = 448 dimensions`

---

## Novel Contributions

1. **First 5-modality fusion for leukemia CDS** — image + lab parameters + clinical text + RAG + LLM in one pipeline
2. **Attention-weighted late fusion** — softmax-normalised learned weights per modality, re-normalised over available modalities
3. **Context-aware multimodal arbitration** — quantitative lab parameters (blast%, WBC) override ambiguous NLP keyword matches; "blast 1%" in a routine CBC does **not** trigger the leukemia flag; "blast 72% AML suspected" correctly does
4. **Graceful modality degradation** — system operates coherently with any subset of modalities (clinically realistic in resource-limited settings)
5. **RAG-grounded diagnosis** — every prediction is cited from peer-reviewed literature (WHO guidelines, NCCN, Hoffbrand's, Blood Journal, JCO, BJH, Leukemia Journal)
6. **Dual XAI** — Grad-CAM highlights suspicious cell regions; SHAP identifies dominant features in the fused numerical vector

---

## Project Structure

```
MedFusion-Leuk/
├── app/
│   ├── api/
│   │   ├── image_route.py      # POST /image/analyze
│   │   ├── nlp_route.py        # POST /nlp/extract
│   │   ├── pdf_route.py        # POST /pdf/extract
│   │   └── predict_route.py    # POST /predict, /predict/image, /predict/full
│   ├── chatbot.py              # MedFusionChatbot (Ollama/meditron)
│   └── main.py                 # FastAPI entry point
│
├── modules/
│   ├── image_analysis/
│   │   ├── model.py            # EfficientNet-B0 + ViT-Tiny hybrid
│   │   ├── gradcam.py          # Grad-CAM implementation
│   │   └── preprocess.py       # Image transforms
│   ├── fusion/
│   │   └── multimodal_fusion.py  # AttentionFusionLayer + parameter confidence
│   ├── nlp/
│   │   └── biobert_ner.py      # BioBERT NER (lazy-loaded)
│   ├── pdf_extraction/
│   │   └── parser.py           # Blood report PDF parser
│   ├── prediction/
│   │   ├── classifier.py       # XGBoost wrapper
│   │   └── risk_scoring.py     # Clinical risk scoring
│   ├── coding/
│   │   └── icd_mapper.py       # ICD-10 + LOINC coding
│   └── rag/
│       ├── embeddings.py       # SentenceTransformer embeddings
│       └── retriever.py        # FAISS retriever
│
├── scripts/
│   ├── prepare_dataset.py              # Step 1: download + split C-NMC
│   ├── train_image_model.py            # Step 2: train EfficientNet-B0 + ViT
│   ├── test_gradcam.py                 # Step 3: Grad-CAM on test images
│   ├── generate_fusion_training_data.py # Step 4: extract fused features
│   ├── train_fusion_classifier.py      # Step 5: train XGBoost + SHAP
│   ├── build_rag_index.py              # Step 6: build FAISS index
│   ├── setup_ollama.py                 # Step 7: pull Ollama model
│   └── start_system.sh                 # Start FastAPI + Streamlit
│
├── evaluation/
│   ├── RESEARCH_RESULTS.md             # Full paper-ready results
│   ├── comparative_results.csv         # Baseline comparison table
│   ├── confusion_matrix.py
│   ├── roc_curve.py
│   ├── comparative_analysis.py
│   └── plots/                          # All generated figures
│
├── frontend/
│   └── streamlit_app.py        # Streamlit UI (Sections A–F)
│
├── tests/                      # 22 pytest tests
├── models/                     # Trained model weights
├── data/                       # Processed features + RAG index
├── config.py
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally

```bash
# 1. Clone the repository
git clone https://github.com/DadaMastan-code/MedFusion-Leuk.git
cd MedFusion-Leuk

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM model (free, local)
ollama pull meditron
```

### Dataset

Download the [C-NMC 2019 dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223) and place it under `data/raw/`. Then run:

```bash
python scripts/prepare_dataset.py
```

The processed features, train/val/test splits, and RAG index are already included in this repository — you can skip directly to running the system if you don't need to retrain.

---

## Quick Start

### Option A — Start everything with one command

```bash
bash scripts/start_system.sh
```

Opens:
- **FastAPI backend** → `http://localhost:8000`
- **API docs (Swagger)** → `http://localhost:8000/docs`
- **Streamlit UI** → `http://localhost:8501`

### Option B — Run servers individually

```bash
# Terminal 1 — FastAPI
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit
streamlit run frontend/streamlit_app.py --server.port 8501
```

---

## API Reference

### `POST /predict` — Text + lab parameters

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_text": "Patient presents with WBC 95000, blast 72%, hemoglobin 7.8. AML suspected.",
    "parameters": {"wbc": 95000, "blast_pct": 72, "hemoglobin": 7.8, "platelets": 42000, "ldh": 800}
  }'
```

**Response schema:**
```json
{
  "prediction":      "ALL",
  "confidence":      0.9998,
  "risk_level":      "Very High",
  "risk_score":      12,
  "risk_triggers":   ["WBC ≥ 50,000", "Blast % ≥ 60%", "..."],
  "icd_code":        "C91.0",
  "icd_description": "Acute lymphoblastic leukemia [ALL]",
  "loinc_codes":     {"6690-2": {"display": "WBC", "value": 95000, "unit": "/µL"}},
  "shap_features":   {"top_features": {"leuk_keyword_flag": 6.43, "ldh": 0.25}},
  "rag_context":     "...",
  "rag_chunks":      [{"source": "WHO Guidelines", "score": 0.725, "text": "..."}],
  "explanation":     "Clinical summary from meditron...",
  "modalities_used": ["parameters", "text"]
}
```

### `POST /predict/image` — Blood smear image upload

```bash
curl -X POST http://localhost:8000/predict/image \
  -F "image=@blood_smear.jpg" \
  -F 'clinical_text=Blast cells noted' \
  -F 'parameters_json={"blast_pct": 72}'
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "llm_model": "meditron", "llm_status": "ready"}
```

---

## Retrain from Scratch

```bash
# Step 1 — Download and split C-NMC 2019
python scripts/prepare_dataset.py

# Step 2 — Train image model (EfficientNet-B0 + ViT-Tiny, ~3h on MPS/GPU)
python scripts/train_image_model.py

# Step 3 — Grad-CAM validation
python scripts/test_gradcam.py

# Step 4 — Generate 448-dim fused features
python scripts/generate_fusion_training_data.py

# Step 5 — Train XGBoost fusion classifier + SHAP
python scripts/train_fusion_classifier.py

# Step 6 — Build FAISS RAG index from literature
python scripts/build_rag_index.py

# Step 7 — Set up Ollama LLM
python scripts/setup_ollama.py
```

---

## Streamlit UI — Sections

| Section | Content |
|---|---|
| **A — Diagnosis Card** | Prediction badge, confidence bar, risk level badge, ICD-10 code |
| **B — Lab Parameters** | Structured table with red highlighting for abnormal values + reference ranges |
| **C — Explainability** | SHAP bar chart (top features) + Grad-CAM overlay |
| **D — Clinical Coding** | ICD-10 code/description, LOINC-coded observations dataframe |
| **E — Literature** | RAG-retrieved chunks with source, similarity score, and text preview |
| **F — AI Chat** | Conversational interface with meditron, session history, clear button |

---

## Testing

```bash
python -m pytest tests/ -v
```

```
22 passed in 5.26s
```

| Test file | Coverage |
|---|---|
| `test_chatbot.py` | Chatbot response, history, context, reset |
| `test_fusion.py` | Multimodal fusion, modality combinations, NaN imputation |
| `test_image_module.py` | Model output shape, transforms, class count |
| `test_nlp_module.py` | Abbreviation expansion, BIO tagging, feature vector |
| `test_pdf_extraction.py` | WBC/Hb/blast parsing, feature shape, normalisation |

---

## Evaluation Plots

All plots are in `evaluation/plots/`:

| Plot | Description |
|---|---|
| `confusion_matrix.png` | Normalised confusion matrix — fusion classifier |
| `roc_curve.png` | ROC-AUC curve (AUC = 1.000) |
| `shap_summary.png` | SHAP feature importance (top 15 features) |
| `training_curves.png` | Train/val loss and accuracy per epoch |
| `comparative_analysis.png` | Baseline comparison bar chart |
| `image_confusion_matrix.png` | Image model confusion matrix |

---

## Clinical Safety Properties

- **No false negatives on high-risk cases** — WBC 95,000 + blast 72%: predicts ALL at 99.98%, Very High risk, 5 risk triggers
- **No false positives on routine CBC** — WBC 7,000 + blast 1%: predicts Normal at 99.95%, Low risk (context-aware arbitration fix)
- **ICD-10 auto-coding** — ALL → C91.0; Normal → Z03.89
- **LOINC annotations** — WBC, Hb, platelet, blast% annotated with standard LOINC codes
- **Mandatory disclaimer** — every LLM explanation includes the clinical decision support disclaimer

---

## Target Journals

| Journal | Impact Factor |
|---|---|
| Computers in Biology and Medicine | ~7.7 |
| Expert Systems with Applications | ~8.5 |
| Applied Soft Computing | ~8.7 |
| IEEE J. Biomedical and Health Informatics | ~7.7 |
| Artificial Intelligence in Medicine | ~7.5 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Image model | PyTorch + timm (EfficientNet-B0 + ViT-Tiny) |
| Fusion classifier | XGBoost + SHAP TreeExplainer |
| Grad-CAM | Custom hook-based PyTorch implementation |
| NLP | Context-aware rule-based (128-dim, BioBERT-format) |
| RAG | FAISS + SentenceTransformer (all-MiniLM-L6-v2) |
| LLM | Ollama / meditron (local, no API key required) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Testing | pytest (22/22) |
| Dataset | C-NMC 2019 (public, CC BY 4.0) |

---

## Citation

If you use MedFusion-Leuk in your research, please cite:

```bibtex
@misc{medfusion_leuk_2026,
  author       = {Mastan, Dada},
  title        = {MedFusion-Leuk: Multimodal Fusion Architecture for Leukemia Clinical Decision Support},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/DadaMastan-code/MedFusion-Leuk}}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The C-NMC 2019 dataset is publicly available under CC BY 4.0 from the Cancer Imaging Archive.

---

> **Clinical Disclaimer:** MedFusion-Leuk is a research prototype and clinical decision *support* tool. It is not a substitute for professional medical diagnosis. All predictions must be reviewed and confirmed by a licensed hematologist before clinical action is taken.
