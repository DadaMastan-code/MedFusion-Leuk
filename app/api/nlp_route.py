"""FastAPI route: clinical text NER via BioBERT."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from modules.nlp.biobert_ner import BioBERTNER

router = APIRouter(prefix="/nlp", tags=["NLP / NER"])

_ner: BioBERTNER | None = None


def _get_ner() -> BioBERTNER:
    global _ner
    if _ner is None:
        _ner = BioBERTNER()
    return _ner


class ClinicalTextRequest(BaseModel):
    text: str


@router.post("/extract-entities")
async def extract_entities(request: ClinicalTextRequest):
    """Run BioBERT NER on clinical text and return entities + feature vector."""
    if not request.text.strip():
        raise HTTPException(400, "Text field must not be empty.")

    ner = _get_ner()
    entities = ner.extract_entities(request.text)
    feature_vec = ner.to_feature_vector(entities)

    return {
        "entities": [
            {"text": e.text, "label": e.label, "score": e.score}
            for e in entities
        ],
        "feature_vector": feature_vec.tolist(),
    }
