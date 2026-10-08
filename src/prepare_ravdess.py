import librosa

# Select one RAVDESS recording
audio_path = "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"

# Load the recording at its original sample rate
audio, original_sr = librosa.load(audio_path, sr=None, mono=True)

print("Original sample rate:", original_sr, "Hz")
print("Original number of samples:", len(audio))

# Resample to 16 kHz for UniSpeech-SAT and WavLM
audio_16k = librosa.resample(
    audio,
    orig_sr=original_sr,
    target_sr=16000
)

print("Resampled sample rate: 16000 Hz")
print("Resampled number of samples:", len(audio_16k))
