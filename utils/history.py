"""
Topic history — tracks every generated episode and enables fuzzy cache lookup.

Storage: output/topic_history.json
  [
    {
      "topic":           "Will AI replace software engineers?",
      "transcript_path": "output/transcript_will_ai_replace_software_engineers.json",
      "generated_at":    "2026-03-27T10:30:00",
      "turns":           21
    },
    ...
  ]
"""

import json
import os
from datetime import datetime

from rapidfuzz import fuzz

from config import OUTPUT_DIR

HISTORY_FILE       = os.path.join(OUTPUT_DIR, "topic_history.json")
SIMILARITY_THRESHOLD = 82   # % — token_set_ratio score above which we treat topics as the same


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(history: list) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_similar(topic: str) -> dict | None:
    """
    Search history for a topic similar to the given one.
    Returns the best matching entry (with an added 'similarity' key) if the
    score meets SIMILARITY_THRESHOLD, otherwise None.

    Uses token_set_ratio so word order and minor rephrasing don't matter.
    """
    history = _load()
    best_score = 0
    best_entry = None

    for entry in history:
        score = fuzz.token_set_ratio(topic.lower(), entry["topic"].lower())
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= SIMILARITY_THRESHOLD:
        return {**best_entry, "similarity": best_score}

    return None


def record(topic: str, transcript_path: str, turns: int) -> None:
    """Append a new episode entry to the history index."""
    history = _load()
    history.append({
        "topic":           topic,
        "transcript_path": transcript_path,
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
        "turns":           turns,
    })
    _save(history)


def list_all() -> list:
    """Return the full history list (newest last)."""
    return _load()
