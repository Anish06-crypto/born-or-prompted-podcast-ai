import io

import pyaudio
import requests
from pydub import AudioSegment

from config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL,
    VOICE_ID_ALEX,
    VOICE_ID_SAM,
    VOICE_SETTINGS,
)

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
CHUNK_SIZE = 4096
TARGET_SAMPLE_RATE = 24000

_VOICE_IDS = {
    "Alex": VOICE_ID_ALEX,
    "Sam": VOICE_ID_SAM,
}


def stream_and_play(text: str, speaker: str) -> None:
    """
    Send text to ElevenLabs TTS, buffer the MP3 response,
    decode with pydub, and play via PyAudio.
    """
    voice_id = _VOICE_IDS.get(speaker)
    if not voice_id:
        raise ValueError(f"No voice ID configured for speaker: {speaker!r}")

    settings = VOICE_SETTINGS.get(speaker, VOICE_SETTINGS["Alex"])

    response = requests.post(
        f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}/stream",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": settings,
        },
        stream=True,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs error {response.status_code}: {response.text[:300]}"
        )

    # Buffer the full MP3 response
    mp3_bytes = b"".join(
        chunk for chunk in response.iter_content(chunk_size=CHUNK_SIZE) if chunk
    )

    # Decode MP3 → normalised PCM (mono, 16-bit, 24 kHz)
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).set_sample_width(2)

    # Play via PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=TARGET_SAMPLE_RATE,
        output=True,
    )
    try:
        stream.write(audio.raw_data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
