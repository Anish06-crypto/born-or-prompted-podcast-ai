VALID_SPEAKERS = {"Alex", "Sam"}
MIN_TURNS = 6


def validate_transcript(data: list) -> list:
    """
    Validate and clean a raw transcript list.
    Raises ValueError with a clear message on any structural problem.
    """
    if not isinstance(data, list):
        raise ValueError(f"Transcript must be a JSON array, got {type(data).__name__}")

    cleaned = []
    for i, turn in enumerate(data):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {i} is not an object: {turn!r}")

        speaker = turn.get("speaker", "").strip()
        text = turn.get("text", "").strip()

        if not speaker:
            raise ValueError(f"Turn {i} is missing 'speaker'")
        if not text:
            raise ValueError(f"Turn {i} is missing 'text'")
        if speaker not in VALID_SPEAKERS:
            raise ValueError(
                f"Turn {i} has unknown speaker {speaker!r}. Must be one of {VALID_SPEAKERS}"
            )

        cleaned.append({"speaker": speaker, "text": text})

    if len(cleaned) < MIN_TURNS:
        raise ValueError(
            f"Transcript too short: {len(cleaned)} turns (minimum {MIN_TURNS})"
        )

    return cleaned
