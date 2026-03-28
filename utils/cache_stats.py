"""
Persistent cache hit/miss counter.

Stored in output/logs/cache_stats.json so it accumulates across all runs.
Lets you analyse how effective the fuzzy topic cache is over time.
"""

import json
import os
from datetime import datetime

_STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "logs", "cache_stats.json")


def _load() -> dict:
    if os.path.exists(_STATS_PATH):
        with open(_STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"hits": 0, "misses": 0, "history": []}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(_STATS_PATH), exist_ok=True)
    with open(_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def record_hit(topic: str, similarity: float) -> None:
    """Call when a cached transcript is replayed instead of generating fresh."""
    data = _load()
    data["hits"] += 1
    data["history"].append({
        "type":       "hit",
        "topic":      topic,
        "similarity": similarity,
        "at":         datetime.now().isoformat(timespec="seconds"),
    })
    _save(data)
    total = data["hits"] + data["misses"]
    print(f"  [cache] Hit  ({similarity}% match) — {data['hits']}/{total} hit rate: {data['hits']/total*100:.1f}%")


def record_miss(topic: str) -> None:
    """Call when no cache match is found and a fresh transcript is generated."""
    data = _load()
    data["misses"] += 1
    data["history"].append({
        "type":  "miss",
        "topic": topic,
        "at":    datetime.now().isoformat(timespec="seconds"),
    })
    _save(data)
    total = data["hits"] + data["misses"]
    print(f"  [cache] Miss — {data['hits']}/{total} hit rate: {data['hits']/total*100:.1f}%")


def get_stats() -> dict:
    data = _load()
    total = data["hits"] + data["misses"]
    return {
        "hits":      data["hits"],
        "misses":    data["misses"],
        "total":     total,
        "hit_rate":  round(data["hits"] / total * 100, 1) if total else 0.0,
    }
