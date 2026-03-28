"""
LLM provider abstraction.

Each provider implements generate_turn() — given a system prompt and
conversation history, returns (text, latency_s) for a single podcast turn.

Currently implemented: GroqProvider (covers all Groq-hosted models).
Stub classes for future providers are included for reference.
"""

import re
import time
from abc import ABC, abstractmethod

from groq import Groq, RateLimitError

from config import GROQ_API_KEYS


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Max tokens per single podcast turn (2–4 sentences ~ 150–300 tokens; 400 gives headroom)
_TURN_MAX_TOKENS = 400

# Models that emit <think> blocks by default — disable reasoning at the API level
# so thinking tokens don't consume the max_tokens budget and leak into the transcript.
_THINKING_MODELS = ("qwen3", "qwq")


class LLMProvider(ABC):
    @abstractmethod
    def generate_turn(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> tuple[str, float, int, int]:
        """
        Generate one podcast turn.

        Parameters
        ----------
        system_prompt : str
            The agent's fixed identity / persona.
        messages : list[dict]
            OpenAI-format chat history (role: user/assistant) representing
            the conversation so far, from this agent's perspective.

        Returns
        -------
        (text, latency_s, prompt_tokens, completion_tokens)
        """


class GroqProvider(LLMProvider):
    """Groq-hosted model. Rotates API keys on 429."""

    def __init__(self, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature

    def generate_turn(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> tuple[str, float, int, int]:
        if not GROQ_API_KEYS:
            raise RuntimeError("No GROQ_API_KEYS configured in .env")

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        is_thinking_model = any(m in self.model.lower() for m in _THINKING_MODELS)
        extra = {"reasoning_effort": "none"} if is_thinking_model else {}

        for i, key in enumerate(GROQ_API_KEYS):
            try:
                client = Groq(api_key=key)
                t0 = time.perf_counter()
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=_TURN_MAX_TOKENS,
                    messages=full_messages,
                    **extra,
                )
                latency_s = time.perf_counter() - t0
                raw = response.choices[0].message.content
                usage = response.usage
                prompt_tokens     = usage.prompt_tokens     if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                return _strip_thinking_tags(raw), latency_s, prompt_tokens, completion_tokens
            except RateLimitError:
                print(f"  [llm] Key {i + 1}/{len(GROQ_API_KEYS)} rate limited for {self.model}, rotating...")

        raise RuntimeError(f"All Groq keys rate limited for model {self.model}.")


# ---------------------------------------------------------------------------
# Future provider stubs (implement when needed)
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """Google Gemini via google-generativeai SDK. Not yet implemented."""

    def generate_turn(self, system_prompt: str, messages: list[dict]) -> tuple[str, float, int, int]:
        raise NotImplementedError("GeminiProvider not yet implemented.")


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible endpoint. Not yet implemented."""

    def generate_turn(self, system_prompt: str, messages: list[dict]) -> tuple[str, float, int, int]:
        raise NotImplementedError("OpenAIProvider not yet implemented.")
