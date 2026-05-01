"""
FAISS-backed RAG retriever over indexed medical literature.

# [NOVELTY] RAG-grounded diagnosis: every LLM response is anchored to
# retrieved PubMed / guideline text, preventing hallucination of clinical facts
# — a capability absent from standalone CNN/NLP leukemia classifiers.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

from config import FAISS_INDEX_PATH, RAG_TOP_K, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from modules.rag.embeddings import MedicalEmbedder


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class MedicalRetriever:
    """Build or load a FAISS index over medical literature chunks."""

    def __init__(self):
        self.embedder = MedicalEmbedder()
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[dict] = []   # {"text": str, "source": str}

    # ── Index construction ──────────────────────────────────────────────────

    def build_from_texts(self, documents: List[dict]) -> None:
        """
        documents: list of {"text": str, "source": str}
        Chunks each document, embeds, and adds to FAISS index.
        """
        self.chunks = []
        chunk_texts = []

        for doc in documents:
            raw = doc["text"]
            source = doc["source"]
            # Sliding window chunking
            for start in range(0, len(raw), RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP):
                chunk = raw[start: start + RAG_CHUNK_SIZE]
                if len(chunk.strip()) < 50:
                    continue
                self.chunks.append({"text": chunk, "source": source})
                chunk_texts.append(chunk)

        embeddings = self.embedder.embed(chunk_texts)  # (N, D)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def save(self, path: Path = FAISS_INDEX_PATH) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.json", "w") as f:
            json.dump(self.chunks, f)

    def load(self, path: Path = FAISS_INDEX_PATH) -> None:
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.json") as f:
            self.chunks = json.load(f)

    # ── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[RetrievedChunk]:
        """Return top-k most relevant literature chunks with citations."""
        if self.index is None:
            return []
        q_emb = self.embedder.embed_single(query).reshape(1, -1)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                text=chunk["text"],
                source=chunk["source"],
                score=float(score),
            ))
        return results

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks as a numbered context block for the LLM."""
        if not chunks:
            return "No relevant literature retrieved."
        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(f"[{i}] Source: {c.source}\n{c.text}")
        return "\n\n".join(lines)
