"""
Shared topic context message injected at the start of every agent's
conversation history. Both Lyra and Cipher receive the same context.
"""


def build_topic_context(
    topic: str,
    position_a_seed: str = "",
    position_b_seed: str = "",
) -> str:
    """
    Build the opening context message given to both agents.
    Seeds (from Reddit top comments) nudge each agent's starting angle.
    """
    lines = [f"Today's topic: {topic}"]

    if position_a_seed:
        lines.append(f"\nLyra's opening angle: {position_a_seed}")
    if position_b_seed:
        lines.append(f"\nCipher's opening angle: {position_b_seed}")

    return "\n".join(lines)
