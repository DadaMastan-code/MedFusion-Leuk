"""
ICD-10 / LOINC / CPT mapping module.
Maps predicted leukemia type + lab tests to standard clinical codes.
"""

from dataclasses import dataclass, asdict
from typing import List

from config import ICD10_MAP, ICD10_DESCRIPTIONS, LOINC_MAP
from modules.pdf_extraction.parser import BloodReportFeatures


@dataclass
class LoincObservation:
    name: str
    loinc_code: str
    display: str
    value: float | None
    unit: str


@dataclass
class CodingReport:
    predicted_class: str
    icd10_code: str
    icd10_description: str
    loinc_observations: List[LoincObservation]

    def to_dict(self) -> dict:
        return asdict(self)


def build_coding_report(
    predicted_class: str,
    features: BloodReportFeatures,
) -> CodingReport:
    """
    Generate a structured coding report from prediction and extracted lab values.
    """
    icd10 = ICD10_MAP.get(predicted_class, "Z03.89")
    description = ICD10_DESCRIPTIONS.get(icd10, "Unknown")

    obs: List[LoincObservation] = []

    _lab_map = [
        ("wbc",       "wbc",       "/µL"),
        ("rbc",       "rbc",       "×10⁶/µL"),
        ("hb",        "hb",        "g/dL"),
        ("platelet",  "platelet",  "/µL"),
        ("blast_pct", "blast",     "%"),
        ("ldh",       "ldh",       "U/L"),
        ("uric_acid", "uric_acid", "mg/dL"),
    ]

    for attr_name, loinc_key, unit in _lab_map:
        val = getattr(features, attr_name, None)
        loinc_entry = LOINC_MAP[loinc_key]
        obs.append(LoincObservation(
            name=attr_name,
            loinc_code=loinc_entry["code"],
            display=loinc_entry["display"],
            value=val,
            unit=unit,
        ))

    return CodingReport(
        predicted_class=predicted_class,
        icd10_code=icd10,
        icd10_description=description,
        loinc_observations=obs,
    )
