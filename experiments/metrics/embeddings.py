"""
Singleton loader for the sentence-transformers embedding model.

Loads `all-MiniLM-L6-v2` once on first call and reuses it across all
metrics in a single process. This avoids repeatedly paying the ~1s model
load cost inside tight metric loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: "SentenceTransformer | None" = None
MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> "SentenceTransformer":
    """Return the shared SentenceTransformer instance, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised embedding vectors.

    Returns an (N, D) float32 array where N == len(texts).
    """
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

    Both vectors are assumed to be L2-normalised (as produced by `embed`),
    so this is just the dot product.
    """
    return float(np.dot(a, b))
