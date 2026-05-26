"""In-memory cosine-similarity retriever over manuscript chunks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .llm import LLMProvider


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms


@dataclass
class ChunkIndex:
    chunks: list[str]
    embeddings: np.ndarray  # (n, dim), unit-normalized

    def search(self, query_embedding: np.ndarray, k: int = 4) -> list[str]:
        q = query_embedding.reshape(-1)
        q = q / (np.linalg.norm(q) or 1)
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top]


def build_index(chunks: list[str], llm: LLMProvider, *, batch_size: int = 32) -> ChunkIndex:
    vectors: list[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        vectors.append(llm.embed(chunks[i : i + batch_size]))
    embeddings = _normalize(np.vstack(vectors))
    return ChunkIndex(chunks=chunks, embeddings=embeddings)


def retrieve(index: ChunkIndex, query: str, llm: LLMProvider, *, k: int = 4) -> list[str]:
    q_emb = llm.embed([query])[0]
    return index.search(q_emb, k=k)
