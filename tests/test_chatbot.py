"""Tests for the Ollama-backed MedFusion chatbot."""

from unittest.mock import patch, MagicMock
from app.chatbot import MedFusionChatbot


MOCK_RESPONSE = {
    "message": {
        "content": (
            "Based on the parameters provided — WBC of 95,000/µL and blast "
            "percentage of 68% — this presentation is consistent with Acute "
            "Myeloid Leukemia (AML). Risk level is Very High. "
            "This is a clinical decision SUPPORT tool; confirm with a hematologist."
        )
    }
}


def _make_chatbot() -> MedFusionChatbot:
    with patch("app.chatbot._resolve_model", return_value="meditron"):
        return MedFusionChatbot()


def test_chatbot_basic_response():
    chatbot = _make_chatbot()
    with patch("ollama.chat", return_value=MOCK_RESPONSE):
        response = chatbot.respond("What does WBC of 95000 indicate?")
    assert response is not None
    assert len(response) > 0
    assert "WBC" in response or "AML" in response or "leukemia" in response.lower()


def test_chatbot_history_accumulates():
    chatbot = _make_chatbot()
    with patch("ollama.chat", return_value=MOCK_RESPONSE):
        chatbot.respond("First question")
        chatbot.respond("Follow-up question")
    assert len(chatbot.history) == 4   # 2 user + 2 assistant turns


def test_chatbot_reset_clears_history():
    chatbot = _make_chatbot()
    with patch("ollama.chat", return_value=MOCK_RESPONSE):
        chatbot.respond("A question")
    assert len(chatbot.history) == 2
    chatbot.reset()
    assert len(chatbot.history) == 0


def test_chatbot_with_diagnostic_context():
    from modules.prediction.risk_scoring import RiskResult
    from modules.coding.icd_mapper import CodingReport, LoincObservation
    from app.chatbot import DiagnosticContext

    risk = RiskResult(score=10, level="Very High", triggered=["WBC ≥ 50,000"])
    coding = CodingReport(
        predicted_class="AML",
        icd10_code="C92.0",
        icd10_description="Acute myeloid leukemia [AML]",
        loinc_observations=[
            LoincObservation("wbc", "6690-2", "Leukocytes", 95000.0, "/µL")
        ],
    )
    ctx = DiagnosticContext(
        predicted_class="AML",
        confidence=0.91,
        risk_result=risk,
        coding_report=coding,
        shap_features={"top_shap_features": {"blast_pct": 0.42, "wbc": 0.31}},
        retrieved_literature="[1] Source: PubMed — AML is characterised by …",
        image_uploaded=True,
        pdf_uploaded=True,
        clinical_text="Patient presents with fatigue and pallor.",
    )

    chatbot = _make_chatbot()
    with patch("ollama.chat", return_value=MOCK_RESPONSE) as mock_chat:
        response = chatbot.respond("Summarise the diagnosis.", diagnostic_context=ctx)

    assert response is not None
    # Verify diagnostic summary was included in the message sent to Ollama
    call_messages = mock_chat.call_args[1]["messages"]
    user_content = call_messages[-1]["content"]
    assert "AML" in user_content
    assert "C92.0" in user_content
    assert "Very High" in user_content
