"""Tests for NLP abbreviation expansion and BIO tagging."""

import numpy as np
from modules.nlp.abbreviation import expand_abbreviations
from modules.nlp.bio_tagger import bio_decode, Entity
from config import NLP_FEATURE_SIZE


def test_abbreviation_expansion():
    text = "Patient has WBC of 80K and Hb of 7.5 g/dL. LDH is elevated."
    expanded = expand_abbreviations(text)
    assert "White Blood Cell" in expanded
    assert "Hemoglobin" in expanded
    assert "Lactate Dehydrogenase" in expanded


def test_bio_decode_single_entity():
    tokens = ["blast", "cells", "observed"]
    labels = ["B-DIAGNOSIS", "I-DIAGNOSIS", "O"]
    scores = [0.95, 0.92, 0.10]
    entities = bio_decode(tokens, labels, scores)
    assert len(entities) == 1
    assert entities[0].label == "DIAGNOSIS"
    assert "blast" in entities[0].text.lower()


def test_bio_decode_multiple_entities():
    tokens = ["WBC", "elevated", ",", "patient", "has", "fatigue"]
    labels = ["B-LAB_VALUE", "O", "O", "O", "O", "B-SYMPTOM"]
    scores = [0.9, 0.1, 0.05, 0.05, 0.05, 0.85]
    entities = bio_decode(tokens, labels, scores)
    labels_found = {e.label for e in entities}
    assert "LAB_VALUE" in labels_found
    assert "SYMPTOM" in labels_found


def test_feature_vector_length():
    from modules.nlp.biobert_ner import BioBERTNER, Entity
    ner = BioBERTNER.__new__(BioBERTNER)  # skip __init__ (no model loaded)
    entities = [
        Entity(text="blast cells", label="DIAGNOSIS", raw_label="Disease_disorder", score=0.9),
        Entity(text="fatigue",     label="SYMPTOM",   raw_label="Sign_symptom",     score=0.8),
    ]
    vec = BioBERTNER.to_feature_vector(ner, entities)
    assert vec.shape == (NLP_FEATURE_SIZE,)
    assert vec.dtype == np.float32
