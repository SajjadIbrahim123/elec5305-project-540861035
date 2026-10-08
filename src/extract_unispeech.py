import time
import librosa
import torch
from transformers import AutoFeatureExtractor, UniSpeechSatModel

# Model and test recording
MODEL_NAME = "microsoft/unispeech-sat-base-plus"
AUDIO_PATH = "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"

# Load the recording and resample it to 16 kHz
audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)

print("Audio sample rate:", sr, "Hz")
print("Number of audio samples:", len(audio))

# Load UniSpeech-SAT Base+
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = UniSpeechSatModel.from_pretrained(MODEL_NAME)

# Freeze the model
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

# Prepare the audio for UniSpeech-SAT
inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

# Extract representations
start_time = time.perf_counter()

with torch.no_grad():
    outputs = model(**inputs)

processing_time = time.perf_counter() - start_time

# Representations across time
frame_embeddings = outputs.last_hidden_state

# Temporal mean pooling -> one embedding for the whole recording
pooled_embedding = frame_embeddings.mean(dim=1)

print("Frame embedding shape:", frame_embeddings.shape)
print("Mean-pooled embedding shape:", pooled_embedding.shape)
print("Processing time:", round(processing_time, 3), "seconds")
print("First 10 embedding values:")
print(pooled_embedding[0, :10])
