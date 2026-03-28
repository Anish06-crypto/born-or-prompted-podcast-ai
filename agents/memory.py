"""
Per-agent episodic memory store — Phase 3.

Each agent accumulates memories across episodes:
  - topic        : what the episode was about
  - stance       : the position the agent took early in that episode
  - key_quote    : a representative line from their later turns
  - outcome      : "agreed" | "disagreed" | "unresolved"
  - episode_date : ISO timestamp

Storage: SQLite at output/agent_memory.db
Retrieval: top-k by topic similarity using rapidfuzz (same strategy as history.py)
"""

import os
import sqlite3
from datetime import datetime

from rapidfuzz import fuzz

from config import OUTPUT_DIR

DB_PATH             = os.path.join(OUTPUT_DIR, "agent_memory.db")
TOP_K               = 3
SIMILARITY_THRESHOLD = 45  # lower than history — broader topic matching is intentional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name   TEXT    NOT NULL,
            topic        TEXT    NOT NULL,
            stance       TEXT    NOT NULL,
            key_quote    TEXT    NOT NULL,
            outcome      TEXT    NOT NULL DEFAULT 'unresolved',
            episode_date TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record(
    agent_name: str,
    topic: str,
    stance: str,
    key_quote: str,
    outcome: str = "unresolved",
) -> None:
    """Persist one memory entry for an agent after an episode completes."""
    conn = _conn()
    conn.execute(
        "INSERT INTO memories (agent_name, topic, stance, key_quote, outcome, episode_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (agent_name, topic, stance, key_quote, outcome, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def retrieve(agent_name: str, topic: str, k: int = TOP_K) -> list[dict]:
    """
    Return the top-k most relevant past memories for this agent on this topic.
    Scored by token_set_ratio — word order and rephrasing don't matter.
    """
    conn = _conn()
    rows = conn.execute(
        "SELECT topic, stance, key_quote, outcome, episode_date "
        "FROM memories WHERE agent_name = ? ORDER BY id DESC",
        (agent_name,),
    ).fetchall()
    conn.close()

    scored = []
    for topic_stored, stance, key_quote, outcome, episode_date in rows:
        score = fuzz.token_set_ratio(topic.lower(), topic_stored.lower())
        if score >= SIMILARITY_THRESHOLD:
            scored.append((score, {
                "topic":        topic_stored,
                "stance":       stance,
                "key_quote":    key_quote,
                "outcome":      outcome,
                "episode_date": episode_date,
            }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**entry, "similarity": score} for score, entry in scored[:k]]


def count_all(agent_name: str) -> int:
    """Return the total number of memories stored for this agent."""
    conn = _conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE agent_name = ?", (agent_name,)
    ).fetchone()
    conn.close()
    return row[0]


def format_memory_context(memories: list[dict]) -> str:
    """
    Format a list of retrieved memories into an injection string for an agent's
    context window. Returns empty string if memories is empty.
    """
    if not memories:
        return ""

    lines = ["[Your memory from past episodes on related topics]"]
    for m in memories:
        date = m["episode_date"][:10]
        lines.append(
            f'- Topic: "{m["topic"]}" ({date})\n'
            f'  Your stance then: {m["stance"]}\n'
            f'  Key quote: "{m["key_quote"]}"\n'
            f'  Outcome: {m["outcome"]}'
        )
    lines.append(
        "Draw on these naturally if relevant — reference a past view, note if you've "
        "shifted position. Don't announce it as 'in my memory'; just weave it in."
    )
    return "\n".join(lines)


def build_memory_context(agent_name: str, topic: str) -> str:
    """Convenience wrapper: retrieve + format in one call."""
    return format_memory_context(retrieve(agent_name, topic))
