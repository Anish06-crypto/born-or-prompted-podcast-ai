"""
Session logger — tracks latency, token usage, content quality proxies, and
UX metrics for every Groq + ElevenLabs call.

Writes structured JSON logs to output/logs/ for later analysis.

Log layout
----------
output/logs/
  session_YYYYMMDD_HHMMSS_<id>.json   ← full per-session detail
  summary.jsonl                        ← one-line-per-session index
"""

import json
import os
import uuid
from datetime import datetime

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "logs")

# Involuntary hesitation sounds — signal uncertainty or thinking out loud
_HEDGING_FILLERS = {
    "uh", "um", "uhm", "hmm", "huh", "uh-huh", "uhh", "umm",
    "like", "you know", "i mean",
}

# Conversational discourse markers — can be stylistic or persona-driven (e.g. Cipher uses "look", "right")
_DISCOURSE_MARKERS = {
    "right", "well", "so", "look",
}


def _count_words(text: str) -> int:
    return len(text.split())


def _count_filler_categories(text: str) -> tuple[int, int]:
    """Returns (hedging_count, marker_count)."""
    lowered = text.lower()
    hedging = sum(lowered.count(f) for f in _HEDGING_FILLERS)
    markers = sum(lowered.count(f) for f in _DISCOURSE_MARKERS)
    return hedging, markers


class SessionLogger:
    def __init__(self, topic: str) -> None:
        self.session_id  = uuid.uuid4().hex[:8]
        self.topic       = topic
        self.started_at  = datetime.now()
        self._groq: dict = {}
        self._turns: list[dict] = []
        self._last_turn_end: datetime | None = None
        self._first_audio_s: float | None = None  # seconds from started_at to first audio
        self._memory: dict = {}  # per-agent memory stats

    # ------------------------------------------------------------------
    # Groq
    # ------------------------------------------------------------------
    def log_groq(
        self,
        *,
        latency_s: float,
        key_index: int,
        retries: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self._groq = {
            "latency_s":         round(latency_s, 3),
            "key_index_used":    key_index,
            "retries":           retries,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        }
        print(
            f"  [log] Groq  latency={latency_s:.2f}s  "
            f"tokens={prompt_tokens}→{completion_tokens}  "
            f"key={key_index + 1}  retries={retries}"
        )

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def log_memory(self, agent_name: str, hits: int, scores: list[float], depth: int) -> None:
        """
        Record episodic memory stats for one agent at episode start.

        hits   — number of relevant past memories retrieved (0 = cold start)
        scores — similarity scores for each retrieved memory (rapidfuzz token_set_ratio)
        depth  — total memories stored for this agent across all episodes
        """
        self._memory[agent_name] = {
            "hits":      hits,
            "top_score": round(max(scores), 1) if scores else None,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
            "scores":    [round(s, 1) for s in scores],
            "depth":     depth,
        }
        cold = "cold start" if hits == 0 else f"{hits} hit(s), top={max(scores):.0f}"
        print(f"  [memory] {agent_name}: {cold}  depth={depth}")

    def finalize_memory_depth(self, agent_name: str, depth_after: int) -> None:
        """Update the memory entry with post-episode depth once memories have been written."""
        if agent_name in self._memory:
            self._memory[agent_name]["depth_after"] = depth_after

    # ------------------------------------------------------------------
    # First audio signal
    # ------------------------------------------------------------------
    def log_first_audio(self) -> None:
        """Call this immediately before the first TTS turn begins playing."""
        self._first_audio_s = round(
            (datetime.now() - self.started_at).total_seconds(), 3
        )
        print(f"  [log] Time-to-first-audio = {self._first_audio_s:.2f}s")

    # ------------------------------------------------------------------
    # TTS turn
    # ------------------------------------------------------------------
    def log_turn(
        self,
        *,
        turn: int,
        speaker: str,
        text: str,
        tts_fetch_s: float,
        playback_s: float,
        gen_latency_s: float = 0.0,
        model: str = "",
    ) -> None:
        now = datetime.now()
        inter_gap_s = 0.0
        if self._last_turn_end is not None:
            inter_gap_s = round((now - self._last_turn_end).total_seconds(), 3)

        word_count                 = _count_words(text)
        hedging_count, marker_count = _count_filler_categories(text)

        entry: dict = {
            "turn":               turn,
            "speaker":            speaker,
            "char_count":         len(text),
            "word_count":         word_count,
            "hedging_count":      hedging_count,
            "hedging_per_100w":   round(hedging_count / word_count * 100, 1) if word_count else 0,
            "marker_count":       marker_count,
            "marker_per_100w":    round(marker_count / word_count * 100, 1) if word_count else 0,
            "tts_fetch_s":        round(tts_fetch_s, 3),
            "playback_s":         round(playback_s, 3),
            "inter_turn_gap_s":   inter_gap_s,
        }
        if gen_latency_s:
            entry["gen_latency_s"] = round(gen_latency_s, 3)
        if model:
            entry["model"] = model

        self._turns.append(entry)
        self._last_turn_end = now
        gen_note = f"  gen={gen_latency_s:.2f}s" if gen_latency_s else ""
        print(
            f"  [log] Turn {turn:>3}  {speaker:<7}"
            f"fetch={tts_fetch_s:.2f}s  play={playback_s:.2f}s  "
            f"gap={inter_gap_s:.2f}s  words={word_count}{gen_note}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, mode: str = "live") -> str:
        """
        Write full session JSON + append one line to summary.jsonl.

        Parameters
        ----------
        mode : str
            "live"   — normal streaming playback run
            "export" — MP3 export run (time_to_first_audio_s will be null;
                       playback_s per turn reflects audio segment duration)

        Returns the log file path.
        """
        completed_at = datetime.now()
        total_s = (completed_at - self.started_at).total_seconds()

        fetch_times    = [t["tts_fetch_s"] for t in self._turns]
        playback_times = [t["playback_s"]   for t in self._turns]
        n = len(self._turns)

        # --- Speaker balance ---
        speakers = {}
        for t in self._turns:
            sp = t["speaker"]
            if sp not in speakers:
                speakers[sp] = {"turns": 0, "words": 0, "hedging": 0, "markers": 0}
            speakers[sp]["turns"]   += 1
            speakers[sp]["words"]   += t["word_count"]
            speakers[sp]["hedging"] += t["hedging_count"]
            speakers[sp]["markers"] += t["marker_count"]

        speaker_stats = {}
        for sp, d in speakers.items():
            w = d["words"]
            speaker_stats[sp] = {
                "turns":               d["turns"],
                "turn_ratio":          round(d["turns"] / n, 3) if n else 0,
                "total_words":         w,
                "avg_words_per_turn":  round(w / d["turns"], 1) if d["turns"] else 0,
                "total_hedging":       d["hedging"],
                "hedging_per_100w":    round(d["hedging"] / w * 100, 1) if w else 0,
                "total_markers":       d["markers"],
                "marker_per_100w":     round(d["markers"] / w * 100, 1) if w else 0,
            }

        tts_summary = {
            "total_turns":      n,
            "total_chars":      sum(t["char_count"]  for t in self._turns),
            "total_words":      sum(t["word_count"]  for t in self._turns),
            "total_fetch_s":    round(sum(fetch_times), 3),
            "avg_fetch_s":      round(sum(fetch_times) / n, 3)    if n else 0,
            "p95_fetch_s":      round(sorted(fetch_times)[int(n * 0.95)] if n >= 2 else (fetch_times[0] if n else 0), 3),
            "min_fetch_s":      round(min(fetch_times), 3)         if n else 0,
            "max_fetch_s":      round(max(fetch_times), 3)         if n else 0,
            "total_playback_s": round(sum(playback_times), 3),
            "avg_playback_s":   round(sum(playback_times) / n, 3)  if n else 0,
        }

        session_data = {
            "session_id":            self.session_id,
            "topic":                 self.topic,
            "mode":                  mode,   # "live" | "export"
            "started_at":            self.started_at.isoformat(timespec="seconds"),
            "completed_at":          completed_at.isoformat(timespec="seconds"),
            "total_duration_s":      round(total_s, 2),
            "time_to_first_audio_s": self._first_audio_s,  # null in export mode
            "groq":                  self._groq,
            "memory":                self._memory,
            "tts_summary":           tts_summary,
            "speaker_stats":         speaker_stats,
            "turns":                 self._turns,
        }

        os.makedirs(_LOG_DIR, exist_ok=True)

        # Full detail log
        filename = (
            f"session_{self.started_at.strftime('%Y%m%d_%H%M%S')}"
            f"_{self.session_id}.json"
        )
        log_path = os.path.join(_LOG_DIR, filename)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        # Summary index (one JSON line per session — easy to load into pandas/excel)
        summary_line = {
            "session_id":            self.session_id,
            "topic":                 self.topic,
            "mode":                  mode,
            "started_at":            session_data["started_at"],
            "total_duration_s":      session_data["total_duration_s"],
            "time_to_first_audio_s": self._first_audio_s,
            "groq_latency_s":        self._groq.get("latency_s"),
            "groq_total_tokens":     self._groq.get("total_tokens"),
            "turns":                 n,
            "avg_tts_fetch_s":       tts_summary["avg_fetch_s"],
            "p95_tts_fetch_s":       tts_summary["p95_fetch_s"],
            "avg_playback_s":        tts_summary["avg_playback_s"],
            "speaker_balance":       {sp: d["turn_ratio"] for sp, d in speaker_stats.items()},
            "memory_hits":           {agent: s["hits"] for agent, s in self._memory.items()},
            "memory_depth":          {agent: s["depth"] for agent, s in self._memory.items()},
        }
        summary_path = os.path.join(_LOG_DIR, "summary.jsonl")
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_line) + "\n")

        print(f"\n  [log] Session saved → {log_path}")
        return log_path
