"""
Experiment runner — builds the full condition matrix and executes conversations.

Usage:
  python -m experiments.runner --dry-run                          # print full matrix, no API calls
  python -m experiments.runner --experiment model_isolation       # run Experiment A
  python -m experiments.runner --experiment persona_isolation     # run Experiment B
  python -m experiments.runner --experiment all                   # run both A then B
  python -m experiments.runner --status                           # show completion progress
  python -m experiments.runner --condition modiso__llama-70b__t03__r02  # run one condition

Checkpoint-resume:
  The runner checks for an existing output file before each condition.
  If the file exists, the condition is skipped. Safe to interrupt and restart.
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime

from agents.generate import generate_transcript
from experiments.conditions import (
    BASELINE_SYSTEM_PROMPT,
    EXPERIMENT_A_MODELS,
    EXPERIMENT_TEMPERATURE,
    FIXED_MODEL_A_B,
    FIXED_MODEL_B,
    FIXED_PROVIDER_A_B,
    FIXED_PROVIDER_B,
    RUNS_PER_CONDITION,
    ExperimentCondition,
)
from experiments.topics import TOPICS


# ── Directory layout ──────────────────────────────────────────────────────────

_BASE_DIR      = os.path.dirname(__file__)
DATA_DIR       = os.path.join(_BASE_DIR, "data")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
METRICS_DIR    = os.path.join(DATA_DIR, "metrics")
RESULTS_DIR    = os.path.join(DATA_DIR, "results")


# ── Rate limiting — conservative sleeps between conversations ─────────────────
# Based on Agent A's provider (the variable one in Experiment A).
# Groq:     30 RPM per model — natural turn latency keeps us safe, buffer for bursts
# Gemini:   10 RPM — 11 Lyra turns per conversation, sleep ensures we don't burst
# Cerebras: 30 RPM but 1M tokens/day — sleep prevents token quota exhaustion

_INTER_CONVERSATION_SLEEP: dict[str, int] = {
    "groq":     4,
    "gemini":   10,
    "cerebras": 6,
}


# ── File helpers ──────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    for d in (TRANSCRIPT_DIR, METRICS_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)


def _transcript_path(condition_id: str) -> str:
    return os.path.join(TRANSCRIPT_DIR, f"{condition_id}.json")


def _is_complete(condition_id: str) -> bool:
    return os.path.exists(_transcript_path(condition_id))


def _save_condition(
    condition: ExperimentCondition,
    transcript: list,
    total_latency: float,
) -> str:
    """Persist one completed condition to experiments/data/transcripts/."""
    payload = {
        "condition":           asdict(condition),
        "transcript":          transcript,
        "generated_at":        datetime.now().isoformat(timespec="seconds"),
        "total_gen_latency_s": round(total_latency, 3),
        "turn_count":          len(transcript),
    }
    path = _transcript_path(condition.condition_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ── Matrix builders ───────────────────────────────────────────────────────────

def build_model_isolation_matrix() -> list[ExperimentCondition]:
    """
    Experiment A — 8 models × 8 topics × 3 runs = 192 conditions.
    Agent A varies across model configurations.
    Agent B (Cipher) is fixed on qwen/qwen3-32b (Groq) throughout.
    """
    conditions: list[ExperimentCondition] = []

    for model_cfg in EXPERIMENT_A_MODELS:
        for t_idx, topic in enumerate(TOPICS):
            for run in range(1, RUNS_PER_CONDITION + 1):
                cid = f"modiso__{model_cfg['slug']}__t{t_idx:02d}__r{run:02d}"
                conditions.append(ExperimentCondition(
                    condition_id             = cid,
                    experiment_tag           = "model_isolation",
                    model_a                  = model_cfg["model"],
                    provider_a               = model_cfg["provider"],
                    model_b                  = FIXED_MODEL_B,
                    provider_b               = FIXED_PROVIDER_B,
                    temperature              = EXPERIMENT_TEMPERATURE,
                    topic                    = topic,
                    topic_index              = t_idx,
                    run_index                = run,
                    persona_slug_a           = model_cfg["slug"],
                    system_prompt_override_a = None,  # Lyra's default prompt throughout
                ))

    return conditions


def build_persona_isolation_matrix() -> list[ExperimentCondition]:
    """
    Experiment B — 3 persona conditions × 8 topics × 3 runs = 72 conditions.
    Agent A's system prompt varies (Lyra / Cipher / Baseline).
    Both models are fixed: llama-3.3-70b (A) and qwen/qwen3-32b (B).
    """
    from agents.personas import AGENTS  # lazy import avoids circular dependency
    lyra_prompt   = AGENTS[0].system_prompt   # Lyra's full persona prompt
    cipher_prompt = AGENTS[1].system_prompt   # Cipher's prompt applied to Agent A slot

    persona_defs: list[tuple[str, str | None]] = [
        ("lyra-persona",   None),                # Lyra's default prompt (no override)
        ("cipher-persona", cipher_prompt),        # Cipher's prompt on Lyra's model
        ("baseline",       BASELINE_SYSTEM_PROMPT),
    ]

    conditions: list[ExperimentCondition] = []

    for slug, prompt_override in persona_defs:
        for t_idx, topic in enumerate(TOPICS):
            for run in range(1, RUNS_PER_CONDITION + 1):
                cid = f"periso__{slug}__t{t_idx:02d}__r{run:02d}"
                conditions.append(ExperimentCondition(
                    condition_id             = cid,
                    experiment_tag           = "persona_isolation",
                    model_a                  = FIXED_MODEL_A_B,
                    provider_a               = FIXED_PROVIDER_A_B,
                    model_b                  = FIXED_MODEL_B,
                    provider_b               = FIXED_PROVIDER_B,
                    temperature              = EXPERIMENT_TEMPERATURE,
                    topic                    = topic,
                    topic_index              = t_idx,
                    run_index                = run,
                    persona_slug_a           = slug,
                    system_prompt_override_a = prompt_override,
                ))

    return conditions


# ── Execution ─────────────────────────────────────────────────────────────────

def run_condition(condition: ExperimentCondition, index: int, total: int) -> bool:
    """
    Execute one condition. Returns True on success, False on failure.
    Prints a clear progress header before and result after.
    """
    print(f"\n{'─' * 62}")
    print(f"  [{index}/{total}]  {condition.condition_id}")
    print(f"  Experiment : {condition.experiment_tag}")
    print(f"  Model A    : {condition.model_a}  ({condition.provider_a})")
    print(f"  Model B    : {condition.model_b}  ({condition.provider_b})")
    print(f"  Persona A  : {condition.persona_slug_a}")
    print(f"  Topic [{condition.topic_index}]  : {condition.topic[:65]}...")
    print(f"  Run        : {condition.run_index}/{RUNS_PER_CONDITION}")
    print(f"{'─' * 62}")

    t0 = time.perf_counter()
    try:
        transcript = generate_transcript(
            condition.topic,
            experiment_mode          = True,
            model_override_a         = condition.model_a,
            provider_override_a      = condition.provider_a,
            model_override_b         = condition.model_b,
            provider_override_b      = condition.provider_b,
            temperature_override     = condition.temperature,
            system_prompt_override_a = condition.system_prompt_override_a,
        )
        total_latency = time.perf_counter() - t0
        path = _save_condition(condition, transcript, total_latency)
        print(f"\n  ✓  {len(transcript)} turns  |  {total_latency:.1f}s  →  {os.path.basename(path)}")
        return True

    except Exception as exc:
        print(f"\n  ✗  FAILED after {time.perf_counter() - t0:.1f}s: {exc}")
        return False


def run_experiment(
    conditions: list[ExperimentCondition],
    dry_run: bool = False,
) -> None:
    """Execute (or preview) a list of conditions with checkpoint-resume."""
    _ensure_dirs()

    total   = len(conditions)
    done    = sum(1 for c in conditions if _is_complete(c.condition_id))
    pending = [c for c in conditions if not _is_complete(c.condition_id)]

    print(f"\n{'═' * 62}")
    print(f"  Total conditions : {total}")
    print(f"  Already complete : {done}")
    print(f"  Pending          : {len(pending)}")
    print(f"{'═' * 62}")

    if dry_run:
        print("\n  [DRY RUN — no API calls will be made]\n")
        for c in pending:
            print(f"  ○  {c.condition_id}")
            print(f"     {c.model_a} ({c.provider_a})  |  topic {c.topic_index}  |  run {c.run_index}")
        return

    if not pending:
        print("\n  All conditions complete — nothing to run.\n")
        return

    errors: list[str] = []
    for i, condition in enumerate(pending, start=done + 1):
        success = run_condition(condition, i, total)
        if not success:
            errors.append(condition.condition_id)

        # Rate-limit buffer between conversations
        if i < total:
            sleep_s = _INTER_CONVERSATION_SLEEP.get(condition.provider_a, 4)
            print(f"  Sleeping {sleep_s}s before next condition...")
            time.sleep(sleep_s)

    print(f"\n{'═' * 62}")
    succeeded = total - done - len(errors)
    print(f"  Run complete.  {succeeded} new  |  {done} prior  |  {len(errors)} failed")
    if errors:
        print(f"\n  Failed conditions:")
        for e in errors:
            print(f"    ✗  {e}")
    print(f"{'═' * 62}\n")


# ── Status display ────────────────────────────────────────────────────────────

def print_status(label: str, conditions: list[ExperimentCondition]) -> None:
    done    = [c for c in conditions if _is_complete(c.condition_id)]
    pending = [c for c in conditions if not _is_complete(c.condition_id)]
    total   = len(conditions)
    pct     = len(done) / total * 100 if total else 0

    print(f"\n  ── {label} ──")
    print(f"  Progress : {len(done)}/{total}  ({pct:.1f}%)")

    if pending:
        preview = pending[:8]
        print(f"  Pending  : {len(pending)} conditions")
        for c in preview:
            print(f"    ○  {c.condition_id}")
        if len(pending) > 8:
            print(f"    ... and {len(pending) - 8} more")

    if done:
        preview = done[-4:]
        print(f"  Recent   : {len(done)} completed")
        for c in preview:
            print(f"    ✓  {c.condition_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Podcast AI — Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.runner --dry-run
  python -m experiments.runner --experiment model_isolation
  python -m experiments.runner --experiment persona_isolation
  python -m experiments.runner --experiment all
  python -m experiments.runner --status
  python -m experiments.runner --condition modiso__llama-70b__t03__r02
        """,
    )
    parser.add_argument(
        "--experiment",
        choices=["model_isolation", "persona_isolation", "all"],
        help="Which experiment matrix to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pending condition list without making any API calls",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print completion progress for both experiments and exit",
    )
    parser.add_argument(
        "--condition",
        type=str,
        metavar="CONDITION_ID",
        help="Run a single condition by its ID (useful for retries and testing)",
    )
    args = parser.parse_args()

    matrix_a = build_model_isolation_matrix()
    matrix_b = build_persona_isolation_matrix()

    # ── Status mode ──
    if args.status:
        _ensure_dirs()
        print_status("Experiment A — Model Isolation",  matrix_a)
        print_status("Experiment B — Persona Isolation", matrix_b)
        total = len(matrix_a) + len(matrix_b)
        done  = sum(1 for c in matrix_a + matrix_b if _is_complete(c.condition_id))
        print(f"\n  Overall: {done}/{total}  ({done/total*100:.1f}%)\n")
        return

    # ── Single condition mode ──
    if args.condition:
        all_conditions = matrix_a + matrix_b
        match = next((c for c in all_conditions if c.condition_id == args.condition), None)
        if not match:
            print(f"\n  Condition not found: {args.condition!r}")
            print("  Use --dry-run to list valid condition IDs.\n")
            return
        _ensure_dirs()
        run_condition(match, index=1, total=1)
        return

    # ── Experiment mode ──
    if not args.experiment:
        parser.print_help()
        return

    if args.experiment in ("model_isolation", "all"):
        print("\n\n══ EXPERIMENT A — Model Isolation ══")
        run_experiment(matrix_a, dry_run=args.dry_run)

    if args.experiment in ("persona_isolation", "all"):
        print("\n\n══ EXPERIMENT B — Persona Isolation ══")
        run_experiment(matrix_b, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
