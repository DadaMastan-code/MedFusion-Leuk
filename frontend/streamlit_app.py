"""MedFusion-Leuk — Streamlit UI (Phase 4)."""

import sys, json, io
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_MODEL
API_BASE = "http://localhost:8000"

# Normal reference ranges for red highlighting
_NORMAL_RANGES = {
    "wbc":       (4500, 11000,   "/µL"),
    "hb":        (12.0, 17.5,    "g/dL"),
    "platelet":  (150000, 400000,"/µL"),
    "blast_pct": (0.0, 1.0,      "%"),
    "ldh":       (140, 280,      "U/L"),
    "uric_acid": (3.5, 7.2,      "mg/dL"),
    "rbc":       (3.8, 5.8,      "×10⁶/µL"),
}

# Risk level colors
_RISK_COLORS = {
    "Low":       "#27ae60",
    "Moderate":  "#f1c40f",
    "High":      "#e67e22",
    "Very High": "#e74c3c",
}
_PRED_COLORS = {"ALL": "#e74c3c", "Normal": "#27ae60"}

st.set_page_config(
    page_title="MedFusion-Leuk",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩸 MedFusion-Leuk")
    st.caption("Multimodal Leukemia Clinical Decision Support")
    st.divider()
    page = st.radio("Navigation", ["Diagnosis", "Evaluation Dashboard", "About"])
    st.divider()

    st.markdown("### System Status")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.ok:
            h = r.json()
            st.success(f"API: online | LLM: {h.get('llm_status','?')}")
        else:
            st.warning("API responding but not healthy")
    except Exception:
        st.error("API offline — start with: uvicorn app.main:app --reload")
    st.divider()
    st.caption(
        "⚠️ Clinical decision **SUPPORT** tool only. "
        "Final diagnosis must be confirmed by a licensed hematologist."
    )


# ── Section A — Diagnosis Card ─────────────────────────────────────────────
def _section_a(result: dict):
    prediction = result.get("prediction", result.get("predicted_class", "?"))
    confidence = result.get("confidence", 0.0)
    risk_level  = result.get("risk_level", "Low")
    icd_code    = result.get("icd_code", result.get("icd10_code", "?"))

    pred_color = _PRED_COLORS.get(prediction, "#7f8c8d")
    risk_color = _RISK_COLORS.get(risk_level, "#7f8c8d")

    st.markdown("## 🔬 Section A — Diagnosis")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"<div style='background:{pred_color};padding:16px;border-radius:10px;"
            f"text-align:center;color:white;font-size:1.6rem;font-weight:bold'>"
            f"{prediction}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='text-align:center;font-size:0.8rem;color:gray'>Prediction</div>",
                    unsafe_allow_html=True)

    with c2:
        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.progress(confidence)

    with c3:
        st.markdown(
            f"<div style='background:{risk_color};padding:16px;border-radius:10px;"
            f"text-align:center;color:white;font-size:1.2rem;font-weight:bold'>"
            f"{risk_level}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:center;font-size:0.8rem;color:gray'>"
            f"Risk Score: {result.get('risk_score', 0)}</div>",
            unsafe_allow_html=True,
        )

    with c4:
        st.metric("ICD-10 Code", icd_code)
        st.caption(result.get("icd_description", result.get("icd10_description", "")))

    triggers = result.get("risk_triggers", [])
    if triggers:
        with st.expander("⚠️ Risk Triggers", expanded=True):
            for t in triggers:
                st.warning(t)
    else:
        st.success("No high-risk thresholds triggered.")


# ── Section B — Clinical Parameters ────────────────────────────────────────
def _section_b(result: dict):
    st.markdown("## 🧪 Section B — Clinical Parameters")
    labs = result.get("extracted_lab_values", {})
    if not any(v is not None for v in labs.values()):
        st.info("No lab values extracted (no PDF or parameters provided).")
        return

    rows = []
    for param, value in labs.items():
        if value is None:
            continue
        info = _NORMAL_RANGES.get(param, (None, None, ""))
        lo, hi, unit = info
        normal = (lo is None) or (lo <= value <= hi)
        status = "✓ Normal" if normal else "⚠ Abnormal"
        range_str = f"{lo}–{hi} {unit}" if lo is not None else "—"
        rows.append({
            "Parameter": param.replace("_", " ").title(),
            "Value": value,
            "Unit": unit,
            "Normal Range": range_str,
            "Status": status,
        })

    df = pd.DataFrame(rows)

    def _highlight(row):
        if "Abnormal" in row["Status"]:
            return ["background-color: #fde8e8"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_highlight, axis=1),
        use_container_width=True,
    )


# ── Section C — SHAP + Grad-CAM ────────────────────────────────────────────
def _section_c(result: dict):
    st.markdown("## 📊 Section C — SHAP & Grad-CAM")
    shap_col, cam_col = st.columns(2)

    with shap_col:
        st.subheader("SHAP Feature Importance")
        shap_data = result.get("shap_features", {})
        feats = shap_data.get("top_features") or shap_data.get("top_shap_features", {})
        if feats:
            names = list(feats.keys())[:10]
            vals  = [feats[n] for n in names]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(names[::-1], vals[::-1], color="#e74c3c")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Top Feature Contributions")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No SHAP data available.")

    with cam_col:
        st.subheader("Grad-CAM Overlay")
        cam_path = result.get("gradcam_path")
        if cam_path and Path(cam_path).exists():
            st.image(cam_path, caption="Attention regions (Grad-CAM)", use_container_width=True)
        else:
            st.info("No image was uploaded — Grad-CAM not generated.")


# ── Section D — Clinical Coding ────────────────────────────────────────────
def _section_d(result: dict):
    st.markdown("## 🏷️ Section D — Clinical Coding")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ICD-10")
        icd = result.get("icd_code", result.get("icd10_code", "—"))
        desc = result.get("icd_description", result.get("icd10_description", "—"))
        st.markdown(f"**Code:** `{icd}`")
        st.markdown(f"**Description:** {desc}")

    with col2:
        st.subheader("LOINC Codes")
        loinc = result.get("loinc_codes", {})
        if loinc:
            rows = [{"LOINC": code, "Test": info["display"],
                     "Value": info["value"], "Unit": info["unit"]}
                    for code, info in loinc.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No LOINC-coded values (no lab parameters extracted).")


# ── Section E — Literature ──────────────────────────────────────────────────
def _section_e(result: dict):
    st.markdown("## 📚 Section E — Retrieved Literature")
    chunks = result.get("rag_chunks", [])
    if chunks:
        for i, ch in enumerate(chunks[:2], 1):
            score = ch.get("score", 0.0)
            source = ch.get("source", "Unknown")
            text = ch.get("text", "")
            with st.expander(f"[{i}] {source}  (similarity: {score:.3f})", expanded=(i == 1)):
                st.markdown(text)
    else:
        rag = result.get("rag_context", result.get("literature_context", ""))
        if rag and rag != "No relevant literature retrieved.":
            st.text(rag[:800])
        else:
            st.info("No literature retrieved (RAG index may be empty).")


# ── Section F — AI Chat ─────────────────────────────────────────────────────
def _section_f(result: dict):
    st.markdown("## 💬 Section F — AI Clinical Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chatbot_ctx" not in st.session_state:
        st.session_state.chatbot_ctx = None

    if result and st.session_state.chatbot_ctx is None:
        explanation = result.get("explanation") or result.get("llm_response", "")
        if explanation:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": explanation}
            )

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.session_state.chatbot_ctx = None
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask the AI assistant a clinical question…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from app.chatbot import MedFusionChatbot
                    bot = MedFusionChatbot()
                    for msg in st.session_state.chat_history[:-1]:
                        bot.history.append(msg)
                    reply = bot.chat(user_input)
                except Exception as e:
                    reply = f"Chatbot error: {e}"
            st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ════════════════════════════════════════════════════════════════
# PAGE: Diagnosis
# ════════════════════════════════════════════════════════════════
if page == "Diagnosis":
    st.header("🩸 Multimodal Leukemia Diagnosis")

    # ── Input Panel ────────────────────────────────────────────
    with st.expander("📥 Input Panel", expanded=True):
        col_img, col_text, col_params = st.columns(3)

        with col_img:
            st.subheader("Blood Smear Image")
            uploaded_image = st.file_uploader("Upload .jpg/.png/.bmp", type=["jpg","jpeg","png","bmp"])
            if uploaded_image:
                st.image(uploaded_image, caption="Uploaded smear", use_container_width=True)

        with col_text:
            st.subheader("Clinical Notes")
            clinical_text = st.text_area(
                "Paste clinical observations or history",
                height=160,
                placeholder="e.g. Patient presents with WBC 85,000/µL, blast 72%, fatigue…",
            )

        with col_params:
            st.subheader("Lab Parameters")
            wbc       = st.number_input("WBC (/µL)",       value=None, placeholder="e.g. 95000", min_value=0.0, step=100.0)
            hb        = st.number_input("Hemoglobin (g/dL)", value=None, placeholder="e.g. 7.8",  min_value=0.0, step=0.1)
            platelets = st.number_input("Platelets (/µL)",  value=None, placeholder="e.g. 42000", min_value=0.0, step=1000.0)
            blast_pct = st.number_input("Blast % ",          value=None, placeholder="e.g. 72",    min_value=0.0, max_value=100.0, step=0.1)
            ldh       = st.number_input("LDH (U/L)",         value=None, placeholder="e.g. 800",   min_value=0.0, step=10.0)

    if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
        has_image  = uploaded_image is not None
        has_text   = bool(clinical_text.strip()) if clinical_text else False
        has_params = any(v is not None for v in [wbc, hb, platelets, blast_pct, ldh])

        if not has_image and not has_text and not has_params:
            st.error("Provide at least one input: image, clinical text, or lab parameters.")
        else:
            with st.spinner("Running multimodal analysis…"):
                try:
                    params = {}
                    if wbc       is not None: params["wbc"]       = wbc
                    if hb        is not None: params["hemoglobin"] = hb
                    if platelets is not None: params["platelets"] = platelets
                    if blast_pct is not None: params["blast_pct"] = blast_pct
                    if ldh       is not None: params["ldh"]       = ldh

                    if has_image:
                        files = {"image": (uploaded_image.name, uploaded_image.getvalue(), "image/jpeg")}
                        data  = {
                            "clinical_text":    clinical_text or "",
                            "parameters_json":  json.dumps(params),
                        }
                        resp = requests.post(f"{API_BASE}/predict/image",
                                             files=files, data=data, timeout=180)
                    else:
                        payload = {"clinical_text": clinical_text or "", "parameters": params}
                        resp    = requests.post(f"{API_BASE}/predict",
                                                json=payload, timeout=180)

                    resp.raise_for_status()
                    result = resp.json()
                    st.session_state["last_result"] = result
                    st.session_state.chat_history = []
                    st.session_state.chatbot_ctx  = None

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Start with: `uvicorn app.main:app --reload`")
                    st.stop()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.stop()

    # ── Results ────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result:
        st.divider()
        _section_a(result)
        st.divider()
        _section_b(result)
        st.divider()
        _section_c(result)
        st.divider()
        _section_d(result)
        st.divider()
        _section_e(result)
        st.divider()
        _section_f(result)
    else:
        st.info("Enter inputs and click **Run Full Analysis** to see results.")


# ════════════════════════════════════════════════════════════════
# PAGE: Evaluation Dashboard
# ════════════════════════════════════════════════════════════════
elif page == "Evaluation Dashboard":
    st.header("Research Metrics Dashboard")

    eval_dir = Path("evaluation/plots")
    imgs = {
        "Training Curves":        eval_dir / "training_curves.png",
        "Confusion Matrix":       eval_dir / "image_confusion_matrix.png",
        "ROC Curves":             eval_dir / "roc_curves.png",
        "SHAP Summary":           eval_dir / "shap_summary.png",
        "Comparative Analysis":   eval_dir / "comparative_analysis.png",
    }

    for title, path in imgs.items():
        if path.exists():
            with st.expander(title, expanded=False):
                st.image(str(path), use_container_width=True)

    st.subheader("Phase 3 Results Summary")
    data = {
        "Method":   ["SVM", "Random Forest", "CNN only", "BioBERT only", "MedFusion-Leuk (ours)"],
        "Accuracy": [0.968, 1.000, 0.948, 1.000, 1.000],
        "F1 Macro": [0.962, 1.000, 0.939, 1.000, 1.000],
        "AUC-ROC":  [0.998, 1.000, 0.983, 1.000, 1.000],
    }
    df = pd.DataFrame(data).set_index("Method")
    st.dataframe(df.style.highlight_max(axis=0, color="#d4efdf"), use_container_width=True)

    st.caption(
        "Image model: Test Acc 94.69% | Best Val 95.62% (epoch 15/15). "
        "Fusion XGBoost: 100% (synthetic features encode class signal — expected for research prototype)."
    )


# ════════════════════════════════════════════════════════════════
# PAGE: About
# ════════════════════════════════════════════════════════════════
elif page == "About":
    st.header("About MedFusion-Leuk")
    st.markdown("""
## Research Novelty

**MedFusion-Leuk** unifies five modalities in a single end-to-end pipeline for leukemia clinical decision support:

| Modality | Module | Architecture |
|---|---|---|
| Blood smear image | EfficientNet-B0 + ViT-Tiny | Dual-encoder with learned soft gating |
| Blood report PDF | pdfplumber + regex | LOINC-coded parameter extraction |
| Clinical text | BioBERT NER | BIO-tagged entity feature vector |
| Medical literature | FAISS RAG | Retrieval-augmented LLM grounding |
| Conversational AI | Ollama (local) | Zero-cost on-device reasoning |

## Architecture

```
Image ──► CNN-ViT ──► img_feat (256d) ──┐
Params ──► normalise ──► pdf_feat (64d)──┤ concat → XGBoost + SHAP → Diagnosis
Text ───► BioBERT NER ──► nlp_feat(128d)─┘
                                          ↓
                                   Risk Score (rule-based)
                                   ICD-10 / LOINC coding
                                   RAG Literature retrieval
                                   Ollama LLM explanation
```

## Image Model Performance
- **Test Accuracy:** 94.69%
- **Best Val Accuracy:** 95.62% (epoch 15/15)
- **Dataset:** C-NMC 2019 — 10,661 images (ALL vs Normal)

## Disclaimer
This system is a **research prototype** for academic publication.
It must not be used as the sole basis for clinical decisions.
""")
