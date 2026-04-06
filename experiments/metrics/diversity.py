"""
Lexical and semantic diversity metrics.

Two complementary measures:

1. Semantic diversity — mean pairwise cosine distance between an agent's
   turn embeddings. High = turns are semantically varied; low = repetitive.

2. Type-token ratio (TTR) — unique word types / total word tokens per agent.
   High = richer vocabulary; low = repetitive word choice.
"""

from __future__ import annotations

import re

import numpy as np

from experiments.metrics.embeddings import embed


def _ttr(texts: list[str]) -> float:
    """Type-token ratio for a flat list of strings."""
    tokens: list[str] = []
    for text in texts:
        tokens.extend(re.findall(r"\b\w+\b", text.lower()))
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """
    Mean pairwise cosine *distance* (1 - similarity) for an (N, D) matrix.
    Returns 0.0 if fewer than 2 turns.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    # embeddings are L2-normalised, so similarity = dot product
    sim_matrix = embeddings @ embeddings.T  # (N, N)
    # extract upper triangle (excluding diagonal)
    upper = sim_matrix[np.triu_indices(n, k=1)]
    distances = 1.0 - upper
    return float(np.mean(distances))


def score_transcript(turns: list[dict]) -> dict:
    """
    Compute diversity metrics per speaker and globally.

    Parameters
    ----------
    turns : list of dicts with keys 'speaker' and 'text'.

    Returns
    -------
    dict with keys:
      'by_speaker' : dict[str, dict] — per-speaker metrics:
                       'semantic_diversity' : mean pairwise cosine distance
                       'ttr'                : type-token ratio
      'global_ttr' : float — TTR across all turns combined
    """
    by_speaker_texts: dict[str, list[str]] = {}
    for turn in turns:
        by_speaker_texts.setdefault(turn["speaker"], []).append(turn["text"])

    by_speaker: dict[str, dict] = {}
    for speaker, texts in by_speaker_texts.items():
        embeddings = embed(texts)
        by_speaker[speaker] = {
            "semantic_diversity": _mean_pairwise_distance(embeddings),
            "ttr": _ttr(texts),
        }

    all_texts = [t["text"] for t in turns]
    return {
        "by_speaker": by_speaker,
        "global_ttr": _ttr(all_texts),
    }
