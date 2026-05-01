"""Sentence-transformer embeddings for RAG vector store."""

from sentence_transformers import SentenceTransformer
import numpy as np
from config import SENTENCE_TRANSFORMER_ID


class MedicalEmbedder:
    """Thin wrapper around SentenceTransformer for medical text."""

    def __init__(self, model_id: str = SENTENCE_TRANSFORMER_ID):
        self.model = SentenceTransformer(model_id)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns (N, D) float32 embeddings."""
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
