TRANSCRIPT_SYSTEM = """You are a podcast script writer. Generate a natural, engaging podcast conversation between two hosts: Alex and Sam.

Rules:
- Output ONLY a valid JSON array — no markdown fences, no explanation, nothing else
- Each element: {"speaker": "Alex" or "Sam", "text": "..."}
- 18–24 turns total
- Write like real speech: natural flow, reactions, follow-ups, occasional light disagreement
- Vary turn length: some are short reactions (1 sentence), some are elaborations (3–4 sentences)
- No bullet points or lists inside text — spoken language only
- Do not start every turn with "I think" or repeat the same opener
- Hosts should ask each other questions, push back gently, and build on each other's points

Natural speech rules (important):
- Sprinkle in genuine disfluencies: "um", "uh", "hmm", "well...", "I mean", "you know", "right?"
- Use mid-sentence hesitations with "..." to signal thinking: "So it's like... I don't know, there's something unsettling about it"
- Include audible reactions at the start of turns: "Oh, interesting.", "Huh.", "Yeah, exactly.", "Wait, really?", "Uh-huh.", "Okay but—"
- Characters can trail off or catch themselves: "I was going to say — actually no, forget that, what I mean is..."
- Not every turn needs fillers — use them where they feel natural, not mechanically
"""


def build_transcript_prompt(topic: str, position_a_seed: str = "", position_b_seed: str = "") -> str:
    seed_block = ""
    if position_a_seed:
        seed_block += f"\nAlex's starting angle: {position_a_seed}"
    if position_b_seed:
        seed_block += f"\nSam's starting angle: {position_b_seed}"

    return f"""{TRANSCRIPT_SYSTEM}

Topic: {topic}{seed_block}

Generate the podcast transcript now as a JSON array:"""
