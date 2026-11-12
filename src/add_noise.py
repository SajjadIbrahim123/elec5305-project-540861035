
import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
NOISE_DIR = Path("data/demand")


def load_noise(environment):
    """Load one DEMAND noise recording."""
    path = NOISE_DIR / environment / "ch01.wav"
    noise, sr = sf.read(path, dtype="float32")

    if sr != SAMPLE_RATE or noise.ndim != 1:
        raise ValueError(f"Expected 16 kHz mono audio: {path}")

    return noise


def add_noise_at_snr(speech, noise, snr_db, start_sample=0):
    """Mix speech and noise at a specified signal-to-noise ratio."""
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)

    if speech.ndim != 1 or noise.ndim != 1:
        raise ValueError("Speech and noise must be mono.")

    # Repeat noise if it is shorter than speech.
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)

    # Select a noise segment matching the speech duration.
    max_start = len(noise) - len(speech)
    start = int(start_sample) % (max_start + 1)
    noise_segment = noise[start:start + len(speech)].copy()
    noise_segment -= np.mean(noise_segment)

    # Calculate RMS amplitudes.
    speech_rms = np.sqrt(np.mean(speech.astype(np.float64) ** 2))
    noise_rms = np.sqrt(np.mean(noise_segment.astype(np.float64) ** 2))

    if speech_rms < 1e-10 or noise_rms < 1e-10:
        raise ValueError("Speech or noise has insufficient energy.")

    # Scale noise to achieve the requested SNR.
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    scaled_noise = noise_segment * (target_noise_rms / noise_rms)

    return (speech + scaled_noise).astype(np.float32)

