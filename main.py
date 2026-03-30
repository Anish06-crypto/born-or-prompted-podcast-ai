#!/usr/bin/env python3
"""
Podcast AI — Entry Point

Usage:
  python main.py                                          # auto-fetch topic from Reddit
  python main.py --topic "AI in healthcare"               # manual topic
  python main.py --sub worldnews                          # Reddit fetch from specific subreddit
  python main.py --transcript output/transcript_foo.json  # replay a saved transcript
  python main.py --topic "AI in healthcare" --fresh       # force regeneration (ignore cache)
  python main.py --history                                # list all past episodes
  python main.py --topic "AI in healthcare" --export      # generate transcript and save as MP3
  python main.py --transcript output/transcript_foo.json --export  # export saved transcript to MP3
  python main.py --export                                 # auto-fetch topic and export as MP3
"""

import argparse
import json
import os
import queue as _queue
import sys
import threading

from agents.generate import TOTAL_TURNS, generate_transcript, generate_transcript_stream
from playback.runner import play_from_queue, play_transcript
from reddit.fetch import fetch_episode_seed
from utils.cache_stats import record_hit, record_miss
from utils.history import find_similar, list_all
from utils.logger import SessionLogger
from visuals.orbs import run_visuals, signal_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Podcast Generator")
    parser.add_argument("--topic",      type=str,       help="Topic to discuss (skips Reddit fetch)")
    parser.add_argument("--sub",        type=str,       help="Subreddit to fetch topic from")
    parser.add_argument("--transcript", type=str,       help="Path to a saved transcript JSON to replay")
    parser.add_argument("--fresh",      action="store_true", help="Ignore cache and regenerate even if a similar topic exists")
    parser.add_argument("--history",    action="store_true", help="Print all past episodes and exit")
    parser.add_argument("--export",     action="store_true", help="Export episode as a timestamped MP3 instead of playing")
    return parser.parse_args()


def _run_playback(transcript: list, logger: SessionLogger | None = None) -> None:
    """Runs in a background thread so the main thread can drive the visuals."""
    play_transcript(transcript, logger=logger)
    if logger is not None:
        logger.save()
    signal_done()


def _print_history() -> None:
    episodes = list_all()
    if not episodes:
        print("No episodes generated yet.")
        return
    print(f"\n{'#':<4} {'Generated':<22} {'Turns':<6} Topic")
    print("-" * 80)
    for i, ep in enumerate(episodes, 1):
        print(f"{i:<4} {ep['generated_at']:<22} {ep['turns']:<6} {ep['topic']}")
    print()


def _load_and_play(transcript_path: str) -> None:
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    threading.Thread(target=_run_playback, args=(transcript,), daemon=True).start()
    run_visuals()


def main() -> None:
    args = parse_args()

    # --- History listing mode ---
    if args.history:
        _print_history()
        return

    # --- Replay mode ---
    if args.transcript:
        with open(args.transcript, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        if args.export:
            from output.export import export_episode
            # Derive a readable topic from the filename (strip prefix/extension)
            base = os.path.splitext(os.path.basename(args.transcript))[0]
            topic_label = base.removeprefix("transcript_").replace("_", " ")
            export_episode(transcript, topic=topic_label)
        else:
            print(f"Replaying: {args.transcript}")
            threading.Thread(target=_run_playback, args=(transcript,), daemon=True).start()
            run_visuals()
        return

    # --- Resolve topic ---
    topic = args.topic
    position_a_seed = ""
    position_b_seed = ""

    if not topic:
        print("Fetching topic from Reddit...")
        seed = fetch_episode_seed(subreddit_name=args.sub)
        if seed:
            topic = seed["topic"]
            position_a_seed = seed["position_a_seed"]
            position_b_seed = seed["position_b_seed"]
        else:
            print("\nReddit fetch failed or returned no results.")
            print("Pass a topic manually:  python main.py --topic \"your topic here\"")
            sys.exit(1)

    # --- Cache lookup ---
    if not args.fresh:
        match = find_similar(topic)
        if match:
            print(f"Cache hit ({match['similarity']}% match): {match['topic']!r}")
            print(f"Generated: {match['generated_at']}  |  {match['turns']} turns")
            record_hit(topic, match["similarity"])

            if args.export:
                from output.export import export_episode
                print("Exporting cached transcript as MP3...\n")
                with open(match["transcript_path"], "r", encoding="utf-8") as f:
                    cached = json.load(f)
                # Create a lightweight logger so the export run is fully captured
                export_logger = SessionLogger(match["topic"])
                export_episode(cached, topic=match["topic"], logger=export_logger)
                export_logger.save(mode="export")
            else:
                print(f"Replaying cached transcript. Use --fresh to regenerate.\n")
                _load_and_play(match["transcript_path"])
            return

    # --- Generate fresh transcript ---
    record_miss(topic)
    logger = SessionLogger(topic)

    if args.export:
        # Blocking generation — no streaming needed, just build the full list then export
        from agents.generate import generate_transcript
        from output.export import export_episode
        print("Generating transcript (blocking for export)...")
        transcript = generate_transcript(topic, position_a_seed, position_b_seed, logger=logger)
        export_episode(transcript, topic=topic, logger=logger)
        logger.save(mode="export")
    else:
        # Pipelined: generate + playback run concurrently
        turn_queue: _queue.Queue = _queue.Queue(maxsize=3)

        def _gen_worker() -> None:
            for turn in generate_transcript_stream(topic, position_a_seed, position_b_seed, logger=logger):
                turn_queue.put(turn)
            turn_queue.put(None)  # sentinel

        def _playback_worker() -> None:
            print("\nStarting playback...\n")
            play_from_queue(turn_queue, TOTAL_TURNS, logger=logger)
            logger.save()
            signal_done()

        threading.Thread(target=_gen_worker, daemon=True).start()
        threading.Thread(target=_playback_worker, daemon=True).start()
        run_visuals()


if __name__ == "__main__":
    main()
