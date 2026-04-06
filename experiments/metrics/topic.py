"""
Topic drift metric.

Measures how far each turn has drifted from the original topic by
computing cosine similarity between each turn embedding and the topic
embedding. Tracks the trajectory over the course of the conversation.
"""

from __future__ import annotations

import numpy as np

from experiments.metrics.embeddings import cosine_similarity, embed


def score_transcript(turns: list[dict], topic: str) -> dict:
    """
    Compute per-turn topic adherence scores.

    Parameters
    ----------
    turns : list of dicts with keys 'speaker' and 'text'.
    topic : the original episode topic string.

    Returns
    -------
    dict with keys:
      'per_turn'        : list[float] — similarity of each turn to the topic
      'mean'            : float — mean adherence across all turns
      'min'             : float — lowest adherence (peak drift)
      'drift_slope'     : float — linear regression slope over the per_turn
                          series; negative = drifting away over time
    """
    topic_vec = embed([topic])[0]
    texts = [t["text"] for t in turns]
    turn_embeddings = embed(texts)

    per_turn = [cosine_similarity(topic_vec, turn_embeddings[i]) for i in range(len(turn_embeddings))]

    slope: float | None = None
    if len(per_turn) >= 2:
        x = np.arange(len(per_turn), dtype=np.float64)
        y = np.array(per_turn, dtype=np.float64)
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])

    return {
        "per_turn": per_turn,
        "mean": float(np.mean(per_turn)),
        "min": float(np.min(per_turn)),
        "drift_slope": slope,
    }
