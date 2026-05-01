"""
LLM conversational chatbot using Ollama (local, free, no API key required).
Primary model: meditron (medical-focused). Fallback: llama3.
"""

import ollama
from dataclasses import dataclass
from typing import List, Optional

from config import (
    OLLAMA_MODEL, OLLAMA_FALLBACK_MODEL, MAX_CONVERSATION_TURNS, SYSTEM_PROMPT
)
from modules.coding.icd_mapper import CodingReport
from modules.prediction.risk_scoring import RiskResult


@dataclass
class DiagnosticContext:
    """Structured summary of all module outputs passed to the LLM."""
    predicted_class: str
    confidence: float
    risk_result: RiskResult
    coding_report: CodingReport
    shap_features: dict
    retrieved_literature: str
    image_uploaded: bool = False
    pdf_uploaded: bool = False
    clinical_text: str = ""


def _resolve_model() -> str:
    """Return meditron if available, otherwise fall back to llama3."""
    try:
        available = [m["model"] for m in ollama.list().get("models", [])]
        for name in available:
            if OLLAMA_MODEL in name:
                return OLLAMA_MODEL
        for name in available:
            if OLLAMA_FALLBACK_MODEL in name:
                return OLLAMA_FALLBACK_MODEL
        # Return whatever is first available, or the primary as default
        return available[0] if available else OLLAMA_MODEL
    except Exception:
        return OLLAMA_MODEL


class MedFusionChatbot:
    """
    Multi-turn conversational chatbot backed by a local Ollama LLM.
    Zero cost — runs entirely on-device.
    """

    def __init__(self):
        self.model = _resolve_model()
        self.history: List[dict] = []

    def _build_diagnostic_summary(self, ctx: DiagnosticContext) -> str:
        shap_str = "\n".join(
            f"  • {k}: {v:.4f}"
            for k, v in ctx.shap_features.get("top_shap_features", {}).items()
        )
        risk_reasons = (
            "\n".join(f"  - {r}" for r in ctx.risk_result.triggered)
            or "  - None triggered"
        )
        return f"""
=== DIAGNOSTIC SUMMARY ===
Predicted Class    : {ctx.predicted_class}
Confidence         : {ctx.confidence * 100:.1f}%
Risk Level         : {ctx.risk_result.level} (score: {ctx.risk_result.score})
ICD-10 Code        : {ctx.coding_report.icd10_code} — {ctx.coding_report.icd10_description}

Risk Factors Triggered:
{risk_reasons}

SHAP Feature Importances (top contributors):
{shap_str}

LOINC-coded Lab Values:
{chr(10).join(f"  [{o.loinc_code}] {o.display}: {o.value} {o.unit}" for o in ctx.coding_report.loinc_observations if o.value is not None)}

Input Modalities Used:
  • Blood smear image : {'Yes' if ctx.image_uploaded else 'No (not provided)'}
  • Blood report PDF  : {'Yes' if ctx.pdf_uploaded else 'No (not provided)'}
  • Clinical text     : {'Yes — ' + ctx.clinical_text[:80] + '…' if ctx.clinical_text else 'No'}

=== RETRIEVED LITERATURE ===
{ctx.retrieved_literature}
""".strip()

    def respond(
        self,
        user_message: str,
        diagnostic_context: Optional[DiagnosticContext] = None,
    ) -> str:
        """
        Send a user message (+ optional diagnostic context) to the local LLM.
        Returns the assistant's response string.
        """
        if diagnostic_context is not None:
            summary = self._build_diagnostic_summary(diagnostic_context)
            content = (
                f"The following diagnostic data was computed by MedFusion-Leuk. "
                f"Please provide a clinical summary and answer my question.\n\n"
                f"{summary}\n\nUser question: {user_message}"
            )
        else:
            content = user_message

        self.history.append({"role": "user", "content": content})

        # Trim to MAX_CONVERSATION_TURNS pairs
        if len(self.history) > MAX_CONVERSATION_TURNS * 2:
            self.history = self.history[-(MAX_CONVERSATION_TURNS * 2):]

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self.history,
            ],
        )

        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def chat(self, message: str) -> str:
        """Convenience alias for respond() without diagnostic context."""
        return self.respond(message)

    def reset(self) -> None:
        self.history = []
