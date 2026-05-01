"""
Rule-based clinical risk scoring: R = Σ(wᵢ · fᵢ) with clinical thresholds.
Operates on BloodReportFeatures and returns a risk level + explanation.
"""

from dataclasses import dataclass
from typing import List, Optional

from config import RISK_THRESHOLDS as T, RISK_WEIGHTS as W, RISK_SCORE_BANDS
from modules.pdf_extraction.parser import BloodReportFeatures


@dataclass
class RiskResult:
    score: int
    level: str            # Low / Moderate / High / Very High
    triggered: List[str]  # human-readable reasons


def compute_risk(features: BloodReportFeatures) -> RiskResult:
    """
    Apply clinical threshold rules to hematological parameters.
    Descriptions are built lazily inside the helpers so None values
    never reach format strings.
    """
    score = 0
    triggered: List[str] = []

    def _check(val: Optional[float], threshold: float, weight: int, desc: str) -> None:
        nonlocal score
        if val is not None and val >= threshold:
            score += weight
            triggered.append(desc.format(val=val))

    def _check_below(val: Optional[float], threshold: float, weight: int, desc: str) -> None:
        nonlocal score
        if val is not None and val < threshold:
            score += weight
            triggered.append(desc.format(val=val))

    # WBC thresholds
    _check(
        features.wbc, T["wbc_high"], W["wbc_high"],
        "WBC {val:,.0f}/µL ≥ 50,000 (high-risk elevation)"
    )
    if features.wbc is not None and features.wbc < T["wbc_high"]:
        _check(
            features.wbc, T["wbc_moderate"], W["wbc_moderate"],
            "WBC {val:,.0f}/µL ≥ 20,000 (moderate elevation)"
        )

    # Blast percentage
    _check(
        features.blast_pct, T["blast_very_high"], W["blast_very_high"],
        "Blast % {val:.1f}% ≥ 60% (very high — acute leukemia threshold)"
    )
    if features.blast_pct is not None and features.blast_pct < T["blast_very_high"]:
        _check(
            features.blast_pct, T["blast_high"], W["blast_high"],
            "Blast % {val:.1f}% ≥ 20% (high — WHO AML criterion)"
        )

    # Haemoglobin
    _check_below(
        features.hb, T["hb_low"], W["hb_low"],
        "Hb {val:.1f} g/dL < 9 (anaemia associated with haematologic malignancy)"
    )

    # Platelets
    _check_below(
        features.platelet, T["platelet_low"], W["platelet_low"],
        "Platelets {val:,.0f}/µL < 80,000 (thrombocytopenia)"
    )

    # LDH
    _check(
        features.ldh, T["ldh_high"], W["ldh_high"],
        "LDH {val:.0f} U/L ≥ 600 (elevated tumour lysis marker)"
    )

    # Determine risk band
    level = "Low"
    for band, threshold in sorted(RISK_SCORE_BANDS.items(), key=lambda x: -x[1]):
        if score >= threshold:
            level = band
            break

    return RiskResult(score=score, level=level, triggered=triggered)
