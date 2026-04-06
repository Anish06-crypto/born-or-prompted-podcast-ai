"""
Conversational coherence metric.

Measures how semantically connected each turn is to the previous turn by
computing cosine similarity between consecutive turn embeddings.

A high score indicates the conversation flows naturally; a low score
indicates abrupt topic shifts or non-sequiturs.
"""

from __future__ import annotations

import numpy as np

from experiments.metrics.embeddings import cosine_similarity, embed


def score_transcript(turns: list[dict]) -> dict:
    """
    Compute per-turn coherence scores for a full transcript.

    Parameters
    ----------
    turns : list of dicts with keys 'speaker' and 'text'.

    Returns
    -------
    dict with keys:
      'per_turn'  : list[float] — coherence score for each turn index >= 1
                    (index 0 has no predecessor; stored as None)
      'mean'      : float — mean of all non-None scores
      'min'       : float — minimum coherence score
    """
    if len(turns) < 2:
        return {"per_turn": [None], "mean": None, "min": None}

    texts = [t["text"] for t in turns]
    embeddings = embed(texts)  # (N, D)

    per_turn: list[float | None] = [None]
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        per_turn.append(sim)

    scores = [s for s in per_turn if s is not None]
    return {
        "per_turn": per_turn,
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
    }
