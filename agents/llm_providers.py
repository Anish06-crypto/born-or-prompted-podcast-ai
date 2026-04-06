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

from config import CEREBRAS_API_KEY, GEMINI_API_KEY, GROQ_API_KEYS


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Max tokens per single podcast turn (2–4 sentences ~ 150–300 tokens; 400 gives headroom)
_TURN_MAX_TOKENS = 400

# Models that require a reasoning_effort parameter.
# Qwen3/QwQ accept "none" (fully disables chain-of-thought).
# gpt-oss-20b requires "low"/"medium"/"high" — use "low" to minimise token
# consumption on thinking while still producing a non-empty response.
_THINKING_MODELS: dict[str, str] = {
    "qwen3":      "none",
    "qwq":        "none",
    "gpt-oss-20b": "low",
}


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
    """Groq-hosted model. Rotates within a dedicated key list on 429."""

    def __init__(self, model: str, temperature: float, api_keys: list[str] | None = None) -> None:
        self.model       = model
        self.temperature = temperature
        # Use the supplied key list; fall back to the global shared pool.
        self._keys       = api_keys if api_keys else GROQ_API_KEYS

    def generate_turn(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> tuple[str, float, int, int]:
        if not self._keys:
            raise RuntimeError("No Groq API keys configured for this provider.")

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        effort = next((v for k, v in _THINKING_MODELS.items() if k in self.model.lower()), None)
        extra  = {"reasoning_effort": effort} if effort is not None else {}

        _RATE_LIMIT_PAUSE   = 60   # seconds to wait before retrying after all keys exhausted
        _RATE_LIMIT_RETRIES = 2   # max full-rotation retries before giving up

        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            for i, key in enumerate(self._keys):
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
                    print(f"  [llm] Key {i + 1}/{len(self._keys)} rate limited for {self.model}, rotating...")

            if attempt < _RATE_LIMIT_RETRIES:
                print(f"  [llm] All keys rate limited for {self.model} — waiting {_RATE_LIMIT_PAUSE}s before retry {attempt + 1}/{_RATE_LIMIT_RETRIES}...")
                time.sleep(_RATE_LIMIT_PAUSE)

        raise RuntimeError(f"All Groq keys rate limited for model {self.model} after {_RATE_LIMIT_RETRIES} retries.")


class GeminiProvider(LLMProvider):
    """
    Google Gemini via the OpenAI-compatible REST endpoint exposed by Google AI Studio.
    Endpoint: https://generativelanguage.googleapis.com/v1beta/openai/
    Free tier: gemini-2.5-flash-lite → 1,000 RPD / 15 RPM.
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, model: str, temperature: float) -> None:
        self.model       = model
        self.temperature = temperature

    def generate_turn(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> tuple[str, float, int, int]:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")

        from openai import OpenAI
        client = OpenAI(api_key=GEMINI_API_KEY, base_url=self._BASE_URL)

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=_TURN_MAX_TOKENS,
            messages=full_messages,
        )
        latency_s = time.perf_counter() - t0

        raw               = response.choices[0].message.content or ""
        usage             = response.usage
        prompt_tokens     = usage.prompt_tokens     if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return _strip_thinking_tags(raw), latency_s, prompt_tokens, completion_tokens


class CerebrasProvider(LLMProvider):
    """
    Cerebras inference via the OpenAI-compatible REST endpoint.
    Endpoint: https://api.cerebras.ai/v1
    Free tier: 1M tokens/day, 30 RPM — primary model: qwen3-235b.
    Note: thinking tags are stripped for Qwen3-series models.
    """

    _BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(self, model: str, temperature: float) -> None:
        self.model       = model
        self.temperature = temperature

    def generate_turn(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> tuple[str, float, int, int]:
        if not CEREBRAS_API_KEY:
            raise RuntimeError("CEREBRAS_API_KEY is not set in .env")

        from openai import OpenAI
        client = OpenAI(api_key=CEREBRAS_API_KEY, base_url=self._BASE_URL)

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=_TURN_MAX_TOKENS,
            messages=full_messages,
        )
        latency_s = time.perf_counter() - t0

        raw               = response.choices[0].message.content or ""
        usage             = response.usage
        prompt_tokens     = usage.prompt_tokens     if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return _strip_thinking_tags(raw), latency_s, prompt_tokens, completion_tokens


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible endpoint. Not yet implemented."""

    def generate_turn(self, system_prompt: str, messages: list[dict]) -> tuple[str, float, int, int]:
        raise NotImplementedError("OpenAIProvider not yet implemented.")
