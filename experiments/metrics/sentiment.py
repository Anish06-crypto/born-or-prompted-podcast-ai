"""
Sentiment metric (VADER).

Computes per-turn compound sentiment scores using VADER, then derives
trajectory slope and volatility across the conversation.

VADER compound score: -1.0 (most negative) to +1.0 (most positive).
"""

from __future__ import annotations

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_transcript(turns: list[dict]) -> dict:
    """
    Compute sentiment metrics for a full transcript.

    Parameters
    ----------
    turns : list of dicts with keys 'speaker' and 'text'.

    Returns
    -------
    dict with keys:
      'per_turn'    : list[float] — VADER compound score per turn
      'by_speaker'  : dict[str, list[float]] — compound scores per speaker
      'mean'        : float — mean compound score across all turns
      'slope'       : float — linear regression slope over the per_turn series
                      (positive = conversation grows more positive over time)
      'volatility'  : float — standard deviation of per-turn scores
                      (high = erratic sentiment swings)
    """
    analyzer = _get_analyzer()
    per_turn: list[float] = []
    by_speaker: dict[str, list[float]] = {}

    for turn in turns:
        score = analyzer.polarity_scores(turn["text"])["compound"]
        per_turn.append(score)
        by_speaker.setdefault(turn["speaker"], []).append(score)

    slope: float | None = None
    if len(per_turn) >= 2:
        x = np.arange(len(per_turn), dtype=np.float64)
        y = np.array(per_turn, dtype=np.float64)
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])

    return {
        "per_turn": per_turn,
        "by_speaker": by_speaker,
        "mean": float(np.mean(per_turn)),
        "slope": slope,
        "volatility": float(np.std(per_turn)),
    }
