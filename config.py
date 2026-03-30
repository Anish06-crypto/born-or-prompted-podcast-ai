import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq ---
# Shared pool — used as fallback if per-agent keys are not set
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]

# Per-agent dedicated keys (comma-separated, same format as GROQ_API_KEYS).
# Agent A (Lyra / llama) uses GROQ_API_KEY_A; Agent B (Cipher / qwen) uses GROQ_API_KEY_B.
# If a per-agent key is absent, the agent falls back to GROQ_API_KEYS.
def _parse_keys(env_var: str) -> list[str]:
    return [k.strip() for k in os.getenv(env_var, "").split(",") if k.strip()]

GROQ_API_KEYS_A = _parse_keys("GROQ_API_KEY_A") or GROQ_API_KEYS
GROQ_API_KEYS_B = _parse_keys("GROQ_API_KEY_B") or GROQ_API_KEYS

# --- Agent models & providers ---
# Provider must match a key in agents/llm_providers.py: "groq", "gemini", "openai"
AGENT_A_MODEL    = os.getenv("AGENT_A_MODEL",    "llama-3.3-70b-versatile")
AGENT_B_MODEL    = os.getenv("AGENT_B_MODEL",    "qwen/qwen3-32b")
AGENT_A_PROVIDER = os.getenv("AGENT_A_PROVIDER", "groq")
AGENT_B_PROVIDER = os.getenv("AGENT_B_PROVIDER", "groq")

# --- ElevenLabs ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL   = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
VOICE_ID_LYRA      = os.getenv("VOICE_ID_LYRA", "")
VOICE_ID_CIPHER    = os.getenv("VOICE_ID_CIPHER", "")

VOICE_SETTINGS = {
    "Lyra":   {"stability": 0.4, "similarity_boost": 0.8,  "style": 0.3, "use_speaker_boost": True},
    "Cipher": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.2, "use_speaker_boost": True},
}

# --- Reddit ---
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "podcast_ai/1.0")

# --- Playback ---
PAUSE_BETWEEN_TURNS = 0.4    # seconds gap between different speakers
PAUSE_SAME_SPEAKER  = 0.15   # seconds gap between consecutive same-speaker turns

# --- Output ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
