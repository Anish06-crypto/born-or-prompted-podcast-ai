import time

from config import PAUSE_BETWEEN_TURNS, PAUSE_SAME_SPEAKER
from tts.stream import stream_and_play
from visuals.orbs import set_active_speaker


def play_transcript(transcript: list) -> None:
    """
    Play each turn of the transcript sequentially via TTS.
    Signals the active speaker to the orb visualizer before/after each turn.
    """
    total = len(transcript)
    prev_speaker = None

    for i, turn in enumerate(transcript):
        speaker = turn["speaker"]
        text = turn["text"]

        preview = text[:70] + ("..." if len(text) > 70 else "")
        print(f"  [{i + 1}/{total}] {speaker}: {preview}")

        set_active_speaker(speaker)
        stream_and_play(text, speaker)
        set_active_speaker(None)

        if i < total - 1:
            pause = PAUSE_SAME_SPEAKER if speaker == prev_speaker else PAUSE_BETWEEN_TURNS
            time.sleep(pause)

        prev_speaker = speaker

    print("\nPlayback complete.")
