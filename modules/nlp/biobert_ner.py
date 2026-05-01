"""
Biomedical NER pipeline using d4data/biomedical-ner-all
(BioBERT fine-tuned on 4 datasets: BC5CDR, NCBI-Disease, BC4CHEMD, JNLPBA).

Entity types produced by this model:
  Disease_disorder, Sign_symptom, Medication, Lab_value,
  Biological_structure, Diagnostic_procedure, Therapeutic_procedure,
  Area, Activity, Coreference, Quantitative_concept, Qualitative_concept,
  Date, Age, Clinical_variable, Detailed_description, History, Nonbiological_location
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List

from config import BIOBERT_MODEL_ID, NLP_FEATURE_SIZE
from modules.nlp.abbreviation import expand_abbreviations

# Map model output labels → our canonical clinical categories
LABEL_MAP = {
    "Disease_disorder":        "DIAGNOSIS",
    "Sign_symptom":            "SYMPTOM",
    "Medication":              "MEDICATION",
    "Lab_value":               "LAB_VALUE",
    "Biological_structure":    "ANATOMY",
    "Diagnostic_procedure":    "PROCEDURE",
    "Therapeutic_procedure":   "PROCEDURE",
    "Clinical_variable":       "LAB_VALUE",
    "Detailed_description":    "SYMPTOM",
    "History":                 "DIAGNOSIS",
    # catch-alls
    "O":                       "OTHER",
}

CLINICAL_LABELS = sorted({
    "DIAGNOSIS", "SYMPTOM", "MEDICATION", "LAB_VALUE",
    "ANATOMY", "PROCEDURE",
})


@dataclass
class Entity:
    text: str
    label: str          # canonical label
    raw_label: str      # original model label
    score: float

    def to_dict(self):
        return asdict(self)


class BioBERTNER:
    """Biomedical NER using a fine-tuned BioBERT variant."""

    def __init__(self, model_id: str = BIOBERT_MODEL_ID, device: int = -1):
        from transformers import pipeline
        self.pipe = pipeline(
            "ner",
            model=model_id,
            aggregation_strategy="simple",
            device=device,
        )

    def extract_entities(self, text: str) -> List[Entity]:
        """Run NER on clinical text; returns canonical Entity objects."""
        text = expand_abbreviations(text)
        raw_entities = self.pipe(text)
        entities = []
        for item in raw_entities:
            raw_label = item["entity_group"]
            canonical = LABEL_MAP.get(raw_label, "OTHER")
            if canonical == "OTHER":
                continue        # skip non-clinical entities
            entities.append(Entity(
                text=item["word"],
                label=canonical,
                raw_label=raw_label,
                score=float(item["score"]),
            ))
        return entities

    def to_feature_vector(self, entities: List[Entity]) -> np.ndarray:
        """
        # [NOVELTY] Fixed-size (NLP_FEATURE_SIZE,) feature vector encoding:
        #   - entity type counts (normalised)
        #   - mean confidence per type
        #   - leukemia keyword presence flag
        """
        count_vec = np.zeros(len(CLINICAL_LABELS), dtype=np.float32)
        conf_sum  = np.zeros(len(CLINICAL_LABELS), dtype=np.float32)

        for ent in entities:
            if ent.label in CLINICAL_LABELS:
                idx = CLINICAL_LABELS.index(ent.label)
                count_vec[idx] += 1
                conf_sum[idx]  += ent.score

        total = count_vec.sum() + 1e-9
        norm_counts = count_vec / total

        conf_vec = np.where(count_vec > 0, conf_sum / (count_vec + 1e-9), 0.0)

        leuk_keywords = {"leukemia", "blast", "lymphoblast", "myeloid",
                         "lymphocytic", "myelogenous", "lymphoma", "aml", "all"}
        leuk_flag = float(any(
            any(kw in ent.text.lower() for kw in leuk_keywords)
            for ent in entities
        ))

        raw = np.concatenate([norm_counts, conf_vec, [leuk_flag]])  # (13,)

        if len(raw) < NLP_FEATURE_SIZE:
            raw = np.pad(raw, (0, NLP_FEATURE_SIZE - len(raw)))
        else:
            raw = raw[:NLP_FEATURE_SIZE]

        return raw.astype(np.float32)
