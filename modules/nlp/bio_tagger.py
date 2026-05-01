"""
BIO tagger: converts raw NER output token-label pairs into structured spans.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float


def bio_decode(tokens: List[str], labels: List[str], scores: List[float]) -> List[Entity]:
    """
    Merge B-/I- BIO spans into Entity objects.
    Handles subword tokens (## prefix from WordPiece) by joining them.
    """
    entities: List[Entity] = []
    current_tokens: List[str] = []
    current_label: str = ""
    current_start: int = 0
    current_scores: List[float] = []

    for i, (token, label, score) in enumerate(zip(tokens, labels, scores)):
        if label.startswith("B-"):
            if current_tokens:
                entities.append(_flush(current_tokens, current_label, current_start, i - 1, current_scores))
            current_tokens = [token]
            current_label = label[2:]
            current_start = i
            current_scores = [score]

        elif label.startswith("I-") and current_tokens and label[2:] == current_label:
            current_tokens.append(token)
            current_scores.append(score)

        else:
            if current_tokens:
                entities.append(_flush(current_tokens, current_label, current_start, i - 1, current_scores))
            current_tokens = []
            current_label = ""
            current_scores = []

    if current_tokens:
        entities.append(_flush(current_tokens, current_label, current_start, len(tokens) - 1, current_scores))

    return entities


def _flush(tokens, label, start, end, scores) -> Entity:
    text = " ".join(t.lstrip("##") if t.startswith("##") else t for t in tokens)
    return Entity(
        text=text,
        label=label,
        start=start,
        end=end,
        score=float(sum(scores) / len(scores)),
    )
