"""
Experiment condition definitions.

Defines the full model and persona configurations for both studies,
and the ExperimentCondition dataclass that identifies one conversation run.

Experiment A — Model Isolation
    Research question: Does model choice affect persona consistency when the
    persona prompt is held constant?
    Fixed  : Persona A (Lyra) | Persona B (Cipher) | Model B (qwen/qwen3-32b)
    Varies : Model A across 8 models spanning 4 providers and 8B–235B scale

Experiment B — Persona Isolation
    Research question: Does having a defined persona produce measurably different
    and more consistent behaviour than no persona constraint?
    Fixed  : Model A (llama-3.3-70b) | Model B (qwen/qwen3-32b)
    Varies : Agent A system prompt across 3 conditions (Lyra / Cipher / Baseline)
"""

from dataclasses import dataclass
from typing import Optional


# ── Shared settings ──────────────────────────────────────────────────────────

EXPERIMENT_TEMPERATURE: float = 0.7   # fixed for all experiment runs
RUNS_PER_CONDITION:     int   = 3     # independent runs per (model × topic) cell

# Fixed Agent B (Cipher) — held constant across Experiment A
FIXED_MODEL_B    = "qwen/qwen3-32b"
FIXED_PROVIDER_B = "groq"


# ── Experiment A — Model configurations ──────────────────────────────────────
# Each entry defines Agent A for one condition block.
# Ordered by approximate parameter scale for clarity.

EXPERIMENT_A_MODELS: list[dict] = [
    {
        "model":    "llama-3.1-8b-instant",
        "provider": "groq",
        "slug":     "llama-8b",
        "note":     "Scale floor — 8B dense, LLaMA 3.1 family",
    },
    {
        "model":    "meta-llama/llama-4-scout-17b-16e-instruct",
        "provider": "groq",
        "slug":     "llama4-scout",
        "note":     "LLaMA 4 MoE — next-generation architecture",
    },
    {
        "model":    "openai/gpt-oss-20b",
        "provider": "groq",
        "slug":     "gpt-oss-20b",
        "note":     "OpenAI OSS 20B — different training lineage",
    },
    {
        "model":    "gemini-2.5-flash-lite",
        "provider": "gemini",
        "slug":     "gemini-flash-lite",
        "note":     "Google Gemini — only cross-provider entry from Google",
    },
    {
        "model":    "qwen/qwen3-32b",
        "provider": "groq",
        "slug":     "qwen-32b",
        "note":     "Same model as Agent B — intra-model persona test",
    },
    {
        "model":    "llama-3.3-70b-versatile",
        "provider": "groq",
        "slug":     "llama-70b",
        "note":     "Current production default for Lyra",
    },
    {
        "model":    "openai/gpt-oss-120b",
        "provider": "groq",
        "slug":     "gpt-oss-120b",
        "note":     "OpenAI OSS 120B — large model, same family as gpt-oss-20b",
    },
    {
        "model":    "qwen-3-235b-a22b-instruct-2507",
        "provider": "cerebras",
        "slug":     "qwen-235b",
        "note":     "Scale ceiling — 235B MoE on Cerebras silicon",
    },
]


# ── Experiment B — Persona configurations ────────────────────────────────────
# Both agents run on the same model — the cleanest isolation design.
# Only Agent A's system prompt varies across the three persona conditions.
#
# Model choice: openai/gpt-oss-120b (production, Groq)
#
# Rationale:
#   llama-3.3-70b was rejected as the fixed model because it is Lyra's native
#   model — its RLHF training likely reinforces warm/collaborative behaviour,
#   which is Lyra's behavioural profile. Using it would compress the measurable
#   gap between the Lyra and Baseline conditions (model natural tendency acts
#   as a confound).
#
#   openai/gpt-oss-120b is preferred because:
#     - Production stable (no deprecation risk over a multi-week run)
#     - Strongest instruction following of all available production models
#       (120B reliably adopts whichever persona it is given)
#     - No native persona bias — different training lineage from both
#       LLaMA (Lyra) and Qwen (Cipher), so baseline truly reflects
#       unconstrained output rather than accidentally mimicking a persona
#     - Present in Experiment A — findings can be directly cross-referenced
#       (Exp A shows its persona consistency score; Exp B shows how it
#       responds to Lyra / Cipher / Baseline prompt conditions)

FIXED_MODEL_A_B    = "openai/gpt-oss-120b"
FIXED_PROVIDER_A_B = "groq"

FIXED_MODEL_B_B    = "openai/gpt-oss-120b"   # same as A — intentional
FIXED_PROVIDER_B_B = "groq"

BASELINE_SYSTEM_PROMPT = (
    "You are an AI participant in a structured discussion. "
    "Engage thoughtfully with the topic and respond directly to what the other participant says. "
    "Keep responses to 2-4 sentences. Be substantive and direct. "
    "Never begin a turn with your own name."
)


# ── ExperimentCondition ───────────────────────────────────────────────────────

@dataclass
class ExperimentCondition:
    """
    Fully describes one conversation run in the study.

    condition_id is deterministic and human-readable:
      Experiment A: modiso__{model_slug}__t{topic_idx:02d}__r{run:02d}
      Experiment B: periso__{persona_slug}__t{topic_idx:02d}__r{run:02d}
    """
    condition_id:             str
    experiment_tag:           str            # "model_isolation" | "persona_isolation"
    model_a:                  str
    provider_a:               str
    model_b:                  str
    provider_b:               str
    temperature:              float
    topic:                    str
    topic_index:              int            # 0–7 (stable index into TOPICS)
    run_index:                int            # 1, 2, or 3
    persona_slug_a:           str           # human label for Agent A's persona/model
    system_prompt_override_a: Optional[str] = None  # None = use Lyra's default prompt
