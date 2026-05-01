"""
Rule-based regex parser for CBC / haematology report values.
Returns a structured dict and a normalised numpy feature vector.
"""

import re
import numpy as np
from typing import Optional
from dataclasses import dataclass, field, asdict

from modules.pdf_extraction.extractor import extract_text


# ── Value ranges used for min-max normalisation ────────────────────────────
_RANGES = {
    "wbc":       (0, 200_000),
    "rbc":       (0, 10),
    "hb":        (0, 25),
    "platelet":  (0, 1_000_000),
    "blast_pct": (0, 100),
    "ldh":       (0, 5000),
    "uric_acid": (0, 30),
}


@dataclass
class BloodReportFeatures:
    wbc:       Optional[float] = None  # /µL
    rbc:       Optional[float] = None  # ×10⁶/µL
    hb:        Optional[float] = None  # g/dL
    platelet:  Optional[float] = None  # /µL
    blast_pct: Optional[float] = None  # %
    ldh:       Optional[float] = None  # U/L
    uric_acid: Optional[float] = None  # mg/dL

    def to_dict(self) -> dict:
        return asdict(self)

    def to_feature_vector(self) -> np.ndarray:
        """
        Returns a (7,) float32 vector with NaN for missing values.
        Values are min-max normalised using clinical reference ranges.
        """
        fields = ["wbc", "rbc", "hb", "platelet", "blast_pct", "ldh", "uric_acid"]
        vec = []
        for f in fields:
            val = getattr(self, f)
            lo, hi = _RANGES[f]
            if val is None:
                vec.append(np.nan)
            else:
                vec.append(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0))
        return np.array(vec, dtype=np.float32)

    def missing_mask(self) -> np.ndarray:
        """Boolean mask: True where value is present."""
        fields = ["wbc", "rbc", "hb", "platelet", "blast_pct", "ldh", "uric_acid"]
        return np.array([getattr(self, f) is not None for f in fields], dtype=bool)


# ── Regex patterns ─────────────────────────────────────────────────────────
_PATTERNS = {
    "wbc": [
        r"(?:WBC|White\s+Blood\s+(?:Cell|Count))[^\d]*(\d[\d,\.]*)\s*(?:/µL|/uL|×10³|x10\^3|K/µL|K/uL|10\^3)?",
    ],
    "rbc": [
        r"(?:RBC|Red\s+Blood\s+(?:Cell|Count))[^\d]*(\d[\d,\.]*)\s*(?:×10⁶|x10\^6|M/µL)?",
    ],
    "hb": [
        r"(?:Hb|Hgb|Hemoglobin|Haemoglobin)[^\d]*(\d[\d,\.]*)\s*(?:g/dL|g/dl)?",
    ],
    "platelet": [
        r"(?:Platelet|PLT|Thrombocyte)[^\d]*(\d[\d,\.]*)\s*(?:/µL|/uL|×10³|K/µL)?",
    ],
    "blast_pct": [
        r"(?:Blast)[^\d]*(\d[\d,\.]*)\s*%",
        r"(\d[\d,\.]*)\s*%\s*(?:Blast|blast)",
    ],
    "ldh": [
        r"(?:LDH|Lactate\s+Dehydrogenase)[^\d]*(\d[\d,\.]*)\s*(?:U/L|IU/L)?",
    ],
    "uric_acid": [
        r"(?:Uric\s+Acid|Urate)[^\d]*(\d[\d,\.]*)\s*(?:mg/dL|mmol/L)?",
    ],
}


def _parse_value(text: str, key: str) -> Optional[float]:
    for pattern in _PATTERNS[key]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(",", "")
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def parse_report(pdf_path: str) -> BloodReportFeatures:
    """Extract hematological parameters from a blood report PDF."""
    text = extract_text(pdf_path)
    features = BloodReportFeatures(
        wbc=_parse_value(text, "wbc"),
        rbc=_parse_value(text, "rbc"),
        hb=_parse_value(text, "hb"),
        platelet=_parse_value(text, "platelet"),
        blast_pct=_parse_value(text, "blast_pct"),
        ldh=_parse_value(text, "ldh"),
        uric_acid=_parse_value(text, "uric_acid"),
    )

    # WBC unit normalisation: if value looks like ×10³ notation, convert to /µL
    if features.wbc is not None and features.wbc < 500:
        features.wbc = features.wbc * 1000

    return features
