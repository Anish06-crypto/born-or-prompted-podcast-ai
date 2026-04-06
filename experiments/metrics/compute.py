"""
Metrics entry point.

Loads a transcript JSON produced by the experiment runner, runs all five
metric modules, and returns (and saves) a unified metrics JSON.

Usage (CLI):
  python -m experiments.metrics.compute experiments/data/transcripts/modiso__llama-70b__t00__r01.json

Usage (as a library):
  from experiments.metrics.compute import compute_metrics
  result = compute_metrics(transcript_path)
"""

from __future__ import annotations

import json
import os
import sys

from experiments.metrics import coherence, diversity, persona, sentiment, topic


METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "metrics")


def compute_metrics(transcript_path: str, save: bool = True) -> dict:
    """
    Load a transcript file and compute all metrics.

    Parameters
    ----------
    transcript_path : absolute or relative path to a condition transcript JSON.
    save            : if True, write metrics JSON to experiments/data/metrics/.

    Returns
    -------
    dict with keys:
      'condition_id' : str
      'topic'        : str
      'turn_count'   : int
      'persona'      : dict  — per-speaker mean + per-turn scores
      'coherence'    : dict  — mean, min, per_turn
      'topic_drift'  : dict  — mean, min, drift_slope, per_turn
      'sentiment'    : dict  — mean, slope, volatility, by_speaker, per_turn
      'diversity'    : dict  — global_ttr, by_speaker (semantic_diversity + ttr)
    """
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    condition   = data["condition"]
    turns       = data["transcript"]
    topic_str   = condition["topic"]
    condition_id = condition["condition_id"]

    # ── Persona ──────────────────────────────────────────────────────────────
    persona_per_speaker  = persona.score_transcript(turns)
    persona_discrimination = persona.discrimination_report(turns)
    persona_result = {
        "by_speaker": {
            speaker: {
                "mean":      sum(scores) / len(scores),
                "per_turn":  scores,
            }
            for speaker, scores in persona_per_speaker.items()
        },
        "discrimination": persona_discrimination,
    }

    # ── Coherence ────────────────────────────────────────────────────────────
    coherence_result = coherence.score_transcript(turns)

    # ── Topic drift ──────────────────────────────────────────────────────────
    topic_result = topic.score_transcript(turns, topic_str)

    # ── Sentiment ────────────────────────────────────────────────────────────
    sentiment_result = sentiment.score_transcript(turns)

    # ── Diversity ────────────────────────────────────────────────────────────
    diversity_result = diversity.score_transcript(turns)

    # ── Assemble ─────────────────────────────────────────────────────────────
    metrics = {
        "condition_id": condition_id,
        "topic":        topic_str,
        "turn_count":   len(turns),
        "persona":      persona_result,
        "coherence":    coherence_result,
        "topic_drift":  topic_result,
        "sentiment":    sentiment_result,
        "diversity":    diversity_result,
    }

    if save:
        os.makedirs(METRICS_DIR, exist_ok=True)
        out_path = os.path.join(METRICS_DIR, f"{condition_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  Metrics saved → {out_path}")

    return metrics


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.metrics.compute <transcript.json> [<transcript.json> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\nProcessing: {path}")
        result = compute_metrics(path)
        print(f"  condition_id : {result['condition_id']}")
        print(f"  turn_count   : {result['turn_count']}")
        print(f"  coherence    : mean={result['coherence']['mean']:.3f}  min={result['coherence']['min']:.3f}")
        print(f"  topic drift  : mean={result['topic_drift']['mean']:.3f}  slope={result['topic_drift']['drift_slope']:.4f}")
        print(f"  sentiment    : mean={result['sentiment']['mean']:.3f}  volatility={result['sentiment']['volatility']:.3f}")
        for speaker, pdata in result["persona"]["by_speaker"].items():
            print(f"  persona [{speaker}] : mean={pdata['mean']:.3f}")
        for speaker, disc in result["persona"]["discrimination"].items():
            print(f"  discrimination [{speaker}] : own={disc['own_mean']:.3f}  cross={disc['cross_mean']:.3f}  gap={disc['gap']:+.3f}")
        for speaker, ddata in result["diversity"]["by_speaker"].items():
            print(f"  diversity [{speaker}] : semantic={ddata['semantic_diversity']:.3f}  ttr={ddata['ttr']:.3f}")


if __name__ == "__main__":
    _main()
