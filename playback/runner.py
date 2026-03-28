import queue
import time
from concurrent.futures import ThreadPoolExecutor

from config import PAUSE_BETWEEN_TURNS, PAUSE_SAME_SPEAKER
from tts.stream import fetch_audio, play_audio
from visuals.orbs import set_active_speaker


def play_from_queue(q: queue.Queue, total: int, logger=None) -> None:
    """
    Play turns from a queue with TTS prefetch.

    While turn N is playing, the next turn's audio is fetched in the background,
    eliminating per-turn fetch latency from the gap between speakers.

    Sentinel: a None value in the queue signals end of stream.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    i = 0

    # Get first turn and immediately start fetching its audio
    turn = q.get()
    if turn is None:
        executor.shutdown(wait=False)
        print("\nPlayback complete.")
        return

    prefetch = executor.submit(fetch_audio, turn["text"], turn["speaker"])

    while turn is not None:
        # Resolve current audio (fetch started before the previous turn played)
        audio, fetch_s = prefetch.result()

        # Get next turn and start pre-fetching its audio before we play current
        # (gen thread has been running in parallel so this is usually instant)
        next_turn = q.get()
        if next_turn is not None:
            prefetch = executor.submit(fetch_audio, next_turn["text"], next_turn["speaker"])

        # Play current turn
        speaker = turn["speaker"]
        text = turn["text"]
        preview = text[:70] + ("..." if len(text) > 70 else "")
        print(f"  [{i + 1}/{total}] {speaker}: {preview}")

        set_active_speaker(speaker)
        if i == 0 and logger is not None:
            logger.log_first_audio()

        playback_s = play_audio(audio)
        set_active_speaker(None)

        if logger is not None:
            logger.log_turn(
                turn=i + 1,
                speaker=speaker,
                text=text,
                tts_fetch_s=fetch_s,
                playback_s=playback_s,
                gen_latency_s=turn.get("gen_latency_s", 0.0),
                model=turn.get("model", ""),
            )

        if next_turn is not None:
            pause = PAUSE_SAME_SPEAKER if speaker == next_turn["speaker"] else PAUSE_BETWEEN_TURNS
            time.sleep(pause)

        turn = next_turn
        i += 1

    executor.shutdown(wait=False)
    print("\nPlayback complete.")


def play_transcript(transcript: list, logger=None) -> None:
    """Play a pre-loaded transcript list (replay mode)."""
    q: queue.Queue = queue.Queue()
    for turn in transcript:
        q.put(turn)
    q.put(None)
    play_from_queue(q, len(transcript), logger=logger)
