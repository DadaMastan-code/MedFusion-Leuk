"""Clinical abbreviation expansion for NLP preprocessing."""

from config import ABBREVIATION_MAP


def expand_abbreviations(text: str) -> str:
    """Replace known clinical abbreviations with full forms."""
    for abbr, full in ABBREVIATION_MAP.items():
        # Word-boundary aware replacement (case-sensitive for clinical accuracy)
        import re
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text
