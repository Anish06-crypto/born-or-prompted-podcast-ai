import io
import time

import pyaudio
import requests
from pydub import AudioSegment

from config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL,
    VOICE_ID_LYRA,
    VOICE_ID_CIPHER,
    VOICE_SETTINGS,
)

ELEVENLABS_BASE    = "https://api.elevenlabs.io/v1"
CHUNK_SIZE         = 4096
TARGET_SAMPLE_RATE = 24000

_VOICE_IDS = {
    "Lyra":   VOICE_ID_LYRA,
    "Cipher": VOICE_ID_CIPHER,
}


def fetch_audio(text: str, speaker: str) -> tuple[AudioSegment, float]:
    """
    Fetch TTS audio from ElevenLabs and decode it.

    Returns
    -------
    (audio, fetch_latency_s)
        audio           : decoded AudioSegment ready for playback
        fetch_latency_s : seconds from request sent to full MP3 buffered and decoded
    """
    voice_id = _VOICE_IDS.get(speaker)
    if not voice_id:
        raise ValueError(f"No voice ID configured for speaker: {speaker!r}")

    settings = VOICE_SETTINGS.get(speaker, VOICE_SETTINGS["Lyra"])

    t_fetch_start = time.perf_counter()
    response = requests.post(
        f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}/stream",
        headers={
            "xi-api-key":   ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",
        },
        json={
            "text":       text,
            "model_id":   ELEVENLABS_MODEL,
            "voice_settings": settings,
        },
        stream=True,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs error {response.status_code}: {response.text[:300]}"
        )

    mp3_bytes = b"".join(
        chunk for chunk in response.iter_content(chunk_size=CHUNK_SIZE) if chunk
    )
    fetch_latency_s = time.perf_counter() - t_fetch_start

    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).set_sample_width(2)

    return audio, fetch_latency_s


def play_audio(audio: AudioSegment) -> float:
    """
    Play a pre-decoded AudioSegment via PyAudio.

    Returns
    -------
    playback_s : seconds spent writing PCM to the audio device
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=TARGET_SAMPLE_RATE,
        output=True,
    )
    t_play_start = time.perf_counter()
    try:
        stream.write(audio.raw_data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    return time.perf_counter() - t_play_start


def stream_and_play(text: str, speaker: str) -> tuple[float, float]:
    """Fetch and immediately play TTS audio. Returns (fetch_latency_s, playback_s)."""
    audio, fetch_s = fetch_audio(text, speaker)
    playback_s = play_audio(audio)
    return fetch_s, playback_s
