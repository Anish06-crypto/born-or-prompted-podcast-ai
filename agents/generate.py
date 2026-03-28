import json
import os

from groq import Groq, RateLimitError

from config import GROQ_API_KEYS, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE, OUTPUT_DIR
from agents.prompts import build_transcript_prompt
from utils.validator import validate_transcript
from utils.history import record as record_in_history


def _call_groq(prompt: str) -> str:
    """Try each Groq key in order; rotate on 429. Raises RuntimeError if all exhausted."""
    if not GROQ_API_KEYS:
        raise RuntimeError(
            "No Groq API keys configured. Set GROQ_API_KEYS in your .env file."
        )

    for i, key in enumerate(GROQ_API_KEYS):
        try:
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=GROQ_MAX_TOKENS,
                temperature=GROQ_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            print(f"  [groq] Key {i + 1}/{len(GROQ_API_KEYS)} rate limited, rotating...")

    raise RuntimeError("All Groq API keys are rate limited. Try again later.")


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences if the model wrapped its output in them."""
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _save_transcript(transcript: list, topic: str) -> str:
    """Persist transcript to output/ and return the file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in " _-" else "" for c in topic)[:50].strip()
    safe = safe.replace(" ", "_").lower()
    path = os.path.join(OUTPUT_DIR, f"transcript_{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return path


def generate_transcript(
    topic: str,
    position_a_seed: str = "",
    position_b_seed: str = "",
) -> list:
    """
    Generate, validate, and save a podcast transcript.
    Retries up to 3 times on JSON / validation failures.
    Returns the validated transcript list.
    """
    prompt = build_transcript_prompt(topic, position_a_seed, position_b_seed)
    print(f"Generating transcript: {topic!r}")

    last_error = None
    for attempt in range(3):
        if attempt > 0:
            print(f"  Retrying... (attempt {attempt + 1}/3)")

        raw = _call_groq(prompt)
        raw = _strip_fences(raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  JSON parse error: {e} — snippet: {raw[:200]!r}")
            continue

        try:
            transcript = validate_transcript(data)
        except ValueError as e:
            last_error = e
            print(f"  Validation error: {e}")
            continue

        path = _save_transcript(transcript, topic)
        record_in_history(topic, path, len(transcript))
        print(f"  {len(transcript)} turns generated. Saved → {path}")
        return transcript

    raise RuntimeError(
        f"Failed to generate a valid transcript after 3 attempts. Last error: {last_error}"
    )
