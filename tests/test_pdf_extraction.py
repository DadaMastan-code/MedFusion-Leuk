"""Tests for PDF extraction and parsing."""

import numpy as np
import pytest
from unittest.mock import patch

from modules.pdf_extraction.parser import parse_report, BloodReportFeatures, _parse_value


SAMPLE_TEXT = """
Patient: John Doe
WBC: 85,000 /µL
RBC: 2.5 x10^6/µL
Hemoglobin: 8.2 g/dL
Platelet: 60,000 /µL
Blast: 72 %
LDH: 850 U/L
Uric Acid: 9.1 mg/dL
"""


def test_parse_wbc():
    val = _parse_value(SAMPLE_TEXT, "wbc")
    assert val == pytest.approx(85000.0)


def test_parse_hb():
    val = _parse_value(SAMPLE_TEXT, "hb")
    assert val == pytest.approx(8.2)


def test_parse_blast():
    val = _parse_value(SAMPLE_TEXT, "blast_pct")
    assert val == pytest.approx(72.0)


def test_feature_vector_shape():
    feat = BloodReportFeatures(wbc=85000, hb=8.2, blast_pct=72)
    vec = feat.to_feature_vector()
    assert vec.shape == (7,)
    assert vec.dtype == np.float32


def test_feature_vector_normalised():
    feat = BloodReportFeatures(wbc=0, rbc=0, hb=0, platelet=0, blast_pct=0, ldh=0, uric_acid=0)
    vec = feat.to_feature_vector()
    assert (vec == 0).all()

    feat2 = BloodReportFeatures(wbc=200_000, rbc=10, hb=25, platelet=1_000_000,
                                 blast_pct=100, ldh=5000, uric_acid=30)
    vec2 = feat2.to_feature_vector()
    assert (vec2 <= 1.0).all()


def test_parse_report_mock():
    with patch("modules.pdf_extraction.parser.extract_text", return_value=SAMPLE_TEXT):
        features = parse_report("dummy.pdf")
    assert features.wbc == pytest.approx(85000.0)
    assert features.hb == pytest.approx(8.2)
    assert features.blast_pct == pytest.approx(72.0)
    assert features.platelet == pytest.approx(60000.0)
