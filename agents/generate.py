"""
Transcript generation — true dual-agent turn-by-turn conversation.

Each agent (Lyra / Cipher) is its own LLM instance with a distinct model,
persona, and conversation perspective. Turns are generated sequentially:
  1. Lyra generates turn 1 (opener — welcomes listeners and introduces topic)
  2. Cipher reads turn 1 and generates turn 2
  3. Lyra reads turns 1–2 and generates turn 3
  … debate continues for TOTAL_TURNS - 2 rounds …
  N. Lyra generates the final turn (closer — thanks Cipher and listeners)
"""

import json
import os
from dataclasses import replace as _dc_replace

from agents.llm_providers import CerebrasProvider, GeminiProvider, GroqProvider, LLMProvider, OpenAIProvider
from agents.memory import count_all as memory_count_all
from agents.memory import format_memory_context
from agents.memory import record as record_memory
from agents.memory import retrieve as memory_retrieve
from agents.personas import AGENTS, AgentPersona
from agents.prompts import build_topic_context
from config import GROQ_API_KEYS_A, GROQ_API_KEYS_B, OUTPUT_DIR
from utils.history import record as record_in_history
from utils.validator import validate_transcript

TOTAL_TURNS = 21  # 1 opener (Lyra) + 19 debate turns + 1 closer (Lyra)

_PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {
    "groq":     GroqProvider,
    "gemini":   GeminiProvider,
    "cerebras": CerebrasProvider,
    "openai":   OpenAIProvider,
}


# Map agent index → dedicated Groq key pool (index 0 = Lyra/Agent A, 1 = Cipher/Agent B)
_GROQ_KEYS_BY_AGENT_INDEX = [GROQ_API_KEYS_A, GROQ_API_KEYS_B]


def _make_provider(agent: AgentPersona) -> LLMProvider:
    cls = _PROVIDER_CLASSES.get(agent.provider.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider {agent.provider!r} for agent {agent.name}. "
            f"Valid options: {list(_PROVIDER_CLASSES)}"
        )
    if cls is GroqProvider:
        agent_index = next((i for i, a in enumerate(AGENTS) if a.name == agent.name), 0)
        return cls(
            model=agent.model,
            temperature=agent.temperature,
            api_keys=_GROQ_KEYS_BY_AGENT_INDEX[agent_index],
        )
    return cls(model=agent.model, temperature=agent.temperature)


def _build_messages_for_agent(
    agent: AgentPersona,
    topic_context: str,
    history: list[dict],
    turn_index: int,
    closing: bool = False,
    memory_context: str = "",
) -> list[dict]:
    """
    Build the OpenAI-format message list for this agent's next turn.

    From each agent's perspective:
      - Their own previous turns  → role: "assistant"
      - The other agent's turns   → role: "user"
    """
    messages: list[dict] = [{"role": "user", "content": topic_context}]

    if memory_context:
        messages.append({"role": "user", "content": memory_context})

    for past in history:
        role = "assistant" if past["speaker"] == agent.name else "user"
        messages.append({"role": role, "content": past["text"]})

    if closing:
        messages.append({
            "role": "user",
            "content": (
                "Wrap up the podcast. Thank your guest Cipher for the conversation and "
                "thank the listeners for tuning in. 2–3 sentences, warm and natural — "
                "no bullet points, no summary of the debate."
            ),
        })
    elif turn_index == 0:
        messages.append({
            "role": "user",
            "content": (
                "Open the podcast. Welcome the listener and introduce the topic naturally. "
                "Keep it to 2–3 sentences — don't over-explain, just set the scene."
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": "Your turn. Respond naturally to what was just said. 2–4 sentences.",
        })

    return messages


def _save_transcript(transcript: list, topic: str) -> str:
    """Persist transcript to output/ and return the file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in " _-" else "" for c in topic)[:50].strip()
    safe = safe.replace(" ", "_").lower()
    path = os.path.join(OUTPUT_DIR, f"transcript_{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return path


def _record_episode_memories(transcript: list[dict], topic: str) -> None:
    """
    Extract per-agent stances and key quotes from the transcript and persist
    them to the episodic memory store.

    Stance  = agent's first debate turn (not the opener/closer).
    Key quote = agent's last turn before the closing.
    Outcome = "unresolved" by default; future versions can infer convergence.
    """
    for agent in AGENTS:
        turns = [t for t in transcript if t["speaker"] == agent.name]
        # Need at least 2 turns to extract a meaningful stance + quote
        if len(turns) < 2:
            continue

        # Skip Lyra's opener (index 0 overall) — it's just a welcome, not a stance
        debate_turns = turns[1:] if agent.name == AGENTS[0].name else turns

        if not debate_turns:
            continue

        stance    = debate_turns[0]["text"][:200]
        # Skip the very last Lyra turn (closing) when picking the key quote
        quote_pool = debate_turns[:-1] if agent.name == AGENTS[0].name and len(debate_turns) > 1 else debate_turns
        key_quote  = quote_pool[-1]["text"][:200]

        record_memory(
            agent_name=agent.name,
            topic=topic,
            stance=stance,
            key_quote=key_quote,
            outcome="unresolved",
        )


def generate_transcript_stream(
    topic: str,
    position_a_seed: str = "",
    position_b_seed: str = "",
    logger=None,
    *,
    model_override_a: str | None = None,
    provider_override_a: str | None = None,
    model_override_b: str | None = None,
    provider_override_b: str | None = None,
    temperature_override: float | None = None,
    system_prompt_override_a: str | None = None,
    system_prompt_override_b: str | None = None,
    experiment_mode: bool = False,
):
    """
    Generator: yields one turn dict at a time as each is produced.

    Allows the playback pipeline to start playing turn 1 while turns 2–21
    are still being generated, eliminating the pre-generation wait.
    Saves the transcript and records history after the final turn is yielded.

    Experiment parameters (keyword-only)
    -------------------------------------
    model_override_a / model_override_b
        Replace the model string for Agent A or B without touching their persona.
    provider_override_a / provider_override_b
        Replace the provider ("groq" | "gemini" | "cerebras") for Agent A or B.
    temperature_override
        Apply a single fixed temperature to both agents (e.g. 0.7 for all runs).
    system_prompt_override_a / system_prompt_override_b
        Replace the system prompt for Agent A or B — used in persona isolation
        experiments to test Cipher's prompt or a neutral baseline on Agent A.
    experiment_mode
        When True:
          - Skips episodic memory retrieval and injection (eliminates cross-run confound)
          - Skips memory recording, history recording, and transcript save to output/
          - Caller is responsible for persisting the returned transcript
    """
    # --- Build active agent list, applying any overrides via dataclass replacement ---
    active_agents = list(AGENTS)

    _overrides_a: dict = {}
    if model_override_a:                        _overrides_a["model"]         = model_override_a
    if provider_override_a:                     _overrides_a["provider"]      = provider_override_a
    if temperature_override is not None:        _overrides_a["temperature"]   = temperature_override
    if system_prompt_override_a:                _overrides_a["system_prompt"] = system_prompt_override_a
    if _overrides_a:
        active_agents[0] = _dc_replace(active_agents[0], **_overrides_a)

    _overrides_b: dict = {}
    if model_override_b:                        _overrides_b["model"]         = model_override_b
    if provider_override_b:                     _overrides_b["provider"]      = provider_override_b
    if temperature_override is not None:        _overrides_b["temperature"]   = temperature_override
    if system_prompt_override_b:                _overrides_b["system_prompt"] = system_prompt_override_b
    if _overrides_b:
        active_agents[1] = _dc_replace(active_agents[1], **_overrides_b)

    providers = {agent.name: _make_provider(agent) for agent in active_agents}

    topic_context = build_topic_context(topic, position_a_seed, position_b_seed)

    # --- Memory retrieval (skipped in experiment_mode to prevent cross-run contamination) ---
    memory_contexts: dict[str, str] = {agent.name: "" for agent in active_agents}
    if not experiment_mode:
        for agent in active_agents:
            memories  = memory_retrieve(agent.name, topic)
            depth     = memory_count_all(agent.name)
            scores    = [m["similarity"] for m in memories]
            memory_contexts[agent.name] = format_memory_context(memories)
            if logger is not None:
                logger.log_memory(agent.name, hits=len(memories), scores=scores, depth=depth)

    history: list[dict] = []
    total_gen_s          = 0.0
    total_prompt_tok     = 0
    total_completion_tok = 0

    mode_tag = " [experiment]" if experiment_mode else ""
    print(f"\nGenerating conversation{mode_tag}: {topic!r}")
    print(f"  {active_agents[0].name}  → {active_agents[0].model}  ({active_agents[0].provider})")
    print(f"  {active_agents[1].name} → {active_agents[1].model}  ({active_agents[1].provider})\n")

    for i in range(TOTAL_TURNS):
        agent    = active_agents[i % 2]
        provider = providers[agent.name]

        closing  = (i == TOTAL_TURNS - 1)
        messages = _build_messages_for_agent(
            agent, topic_context, history, i,
            closing=closing,
            memory_context=memory_contexts[agent.name],
        )
        text, latency_s, prompt_tok, completion_tok = provider.generate_turn(agent.system_prompt, messages)
        total_gen_s          += latency_s
        total_prompt_tok     += prompt_tok
        total_completion_tok += completion_tok

        turn = {
            "speaker":       agent.name,
            "text":          text,
            "model":         agent.model,
            "gen_latency_s": round(latency_s, 3),
        }
        history.append(turn)
        print(f"  [{i + 1}/{TOTAL_TURNS}] {agent.name} ({latency_s:.2f}s): {text[:65]}...")
        yield turn

    # --- Post-generation: validate, persist, record (all skipped in experiment_mode) ---
    validated = validate_transcript(history)

    if logger is not None:
        logger.log_groq(
            latency_s=total_gen_s,
            key_index=0,
            retries=0,
            prompt_tokens=total_prompt_tok,
            completion_tokens=total_completion_tok,
        )

    if not experiment_mode:
        path = _save_transcript(validated, topic)
        record_in_history(topic, path, len(validated))
        _record_episode_memories(validated, topic)
        if logger is not None:
            for agent in active_agents:
                logger.finalize_memory_depth(agent.name, memory_count_all(agent.name))
        print(f"\n  {len(validated)} turns saved → {path}")
    else:
        print(f"\n  {len(validated)} turns generated (experiment mode — caller handles persistence)")


def generate_transcript(
    topic: str,
    position_a_seed: str = "",
    position_b_seed: str = "",
    logger=None,
    *,
    model_override_a: str | None = None,
    provider_override_a: str | None = None,
    model_override_b: str | None = None,
    provider_override_b: str | None = None,
    temperature_override: float | None = None,
    system_prompt_override_a: str | None = None,
    system_prompt_override_b: str | None = None,
    experiment_mode: bool = False,
) -> list:
    """Blocking version: generates all turns and returns the full transcript list."""
    return list(generate_transcript_stream(
        topic, position_a_seed, position_b_seed, logger=logger,
        model_override_a=model_override_a,
        provider_override_a=provider_override_a,
        model_override_b=model_override_b,
        provider_override_b=provider_override_b,
        temperature_override=temperature_override,
        system_prompt_override_a=system_prompt_override_a,
        system_prompt_override_b=system_prompt_override_b,
        experiment_mode=experiment_mode,
    ))
