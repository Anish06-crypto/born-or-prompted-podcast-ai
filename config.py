import os
from dotenv import load_dotenv

load_dotenv()

# --- Groq ---
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 8192
GROQ_TEMPERATURE = 0.85

# --- ElevenLabs ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
VOICE_ID_ALEX = os.getenv("VOICE_ID_ALEX", "")
VOICE_ID_SAM = os.getenv("VOICE_ID_SAM", "")

VOICE_SETTINGS = {
    "Alex": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.3, "use_speaker_boost": True},
    "Sam":  {"stability": 0.5, "similarity_boost": 0.75, "style": 0.2, "use_speaker_boost": True},
}

# --- Reddit ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "podcast_ai/1.0")

# --- Playback ---
PAUSE_BETWEEN_TURNS = 0.4   # seconds gap between different speakers
PAUSE_SAME_SPEAKER = 0.15   # seconds gap between consecutive same-speaker turns

# --- Output ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
