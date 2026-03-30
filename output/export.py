"""
output/export.py — Export a podcast episode as a timestamped MP3.

Usage (CLI):
    python main.py --topic "AI ethics" --export
    python main.py --transcript output/transcript_foo.json --export
    python main.py --export          # auto-fetch topic from Reddit, then export

Usage (Python API):
    from output.export import export_episode
    path = export_episode(transcript, topic)
"""

import os
import re
from datetime import datetime
from typing import TYPE_CHECKING

from pydub import AudioSegment

from config import OUTPUT_DIR, PAUSE_BETWEEN_TURNS, PAUSE_SAME_SPEAKER
from tts.stream import fetch_audio

if TYPE_CHECKING:
    from utils.logger import SessionLogger


def _slug(topic: str, max_len: int = 50) -> str:
    """Convert a topic string into a safe, lowercase filename slug."""
    slug = topic.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)          # strip punctuation
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")  # collapse whitespace
    return slug[:max_len]


def export_episode(
    transcript: list[dict],
    topic: str,
    logger: "SessionLogger | None" = None,
) -> str:
    """
    Fetch TTS audio for every turn, concatenate with inter-turn silence,
    and export as a timestamped MP3 file into OUTPUT_DIR.

    Parameters
    ----------
    transcript : list[dict]
        Each dict must have at least 'speaker' and 'text' keys.
        (Same format produced by generate_transcript / saved JSON files.)
    topic : str
        Human-readable topic — used to build the output filename.
    logger : SessionLogger | None
        If provided, log_turn() is called after each TTS fetch so the
        session log captures full TTS / speaker stats for export runs.

    Returns
    -------
    str
        Absolute path to the saved MP3 file.
    """
    if not transcript:
        raise ValueError("Transcript is empty — nothing to export.")

    total = len(transcript)
    print(f"\n🎙  Exporting episode  ({total} turns)  [mode: export]")
    print(f"    Topic : {topic}\n")

    combined: AudioSegment = AudioSegment.empty()

    for i, turn in enumerate(transcript):
        speaker = turn["speaker"]
        text    = turn["text"]
        preview = text[:65] + ("..." if len(text) > 65 else "")
        print(f"  [{i + 1:>2}/{total}] {speaker}: {preview}")

        audio, fetch_s = fetch_audio(text, speaker)
        segment_duration_s = len(audio) / 1000  # pydub length is in ms
        print(f"           ↳ fetched in {fetch_s:.2f}s  |  {segment_duration_s:.1f}s audio")

        combined += audio

        # Log this turn to the session logger (playback_s = segment duration)
        if logger is not None:
            logger.log_turn(
                turn=i + 1,
                speaker=speaker,
                text=text,
                tts_fetch_s=fetch_s,
                playback_s=segment_duration_s,
                gen_latency_s=turn.get("gen_latency_s", 0.0),
                model=turn.get("model", ""),
            )

        # Append inter-turn silence (skip after the final turn)
        if i < total - 1:
            next_speaker = transcript[i + 1]["speaker"]
            gap_s = PAUSE_SAME_SPEAKER if speaker == next_speaker else PAUSE_BETWEEN_TURNS
            combined += AudioSegment.silent(duration=int(gap_s * 1000))  # pydub = ms

    # Build output path: episode_YYYYMMDD_HHMMSS_<slug>.mp3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"episode_{timestamp}_{_slug(topic)}.mp3"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path  = os.path.join(OUTPUT_DIR, filename)

    duration_s = len(combined) / 1000
    print(f"\n  Writing MP3 → {out_path}")
    combined.export(out_path, format="mp3", bitrate="128k")

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✓ Saved  |  Duration: {duration_s:.1f}s ({duration_s / 60:.1f} min)  |  Size: {size_kb:.0f} KB")
    print(f"\n  {out_path}\n")

    return out_path
