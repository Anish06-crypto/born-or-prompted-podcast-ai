from agents.personas import AGENTS

VALID_SPEAKERS = {agent.name for agent in AGENTS}
MIN_TURNS = 6


def validate_transcript(data: list) -> list:
    """
    Validate and clean a raw transcript list.
    Raises ValueError with a clear message on any structural problem.
    Preserves extra fields (model, gen_latency_s) added by the generator.
    """
    if not isinstance(data, list):
        raise ValueError(f"Transcript must be a list, got {type(data).__name__}")

    cleaned = []
    for i, turn in enumerate(data):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {i} is not a dict: {turn!r}")

        speaker = turn.get("speaker", "").strip()
        text    = turn.get("text", "").strip()

        if not speaker:
            raise ValueError(f"Turn {i} missing 'speaker'")
        if not text:
            raise ValueError(f"Turn {i} missing 'text'")
        if speaker not in VALID_SPEAKERS:
            raise ValueError(
                f"Turn {i} unknown speaker {speaker!r}. Expected one of {VALID_SPEAKERS}"
            )

        # Preserve all fields from the generator (model, gen_latency_s, etc.)
        entry = {k: v for k, v in turn.items()}
        entry["speaker"] = speaker
        entry["text"]    = text
        cleaned.append(entry)

    if len(cleaned) < MIN_TURNS:
        raise ValueError(f"Transcript too short: {len(cleaned)} turns (min {MIN_TURNS})")

    return cleaned
