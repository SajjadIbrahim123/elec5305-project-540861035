import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, WavLMModel


MODEL_NAME = "microsoft/wavlm-base-plus"
METADATA_PATH = "results/ravdess_metadata.csv"
OUTPUT_PATH = "embeddings/wavlm/embeddings.npy"


# Load dataset metadata
metadata = pd.read_csv(METADATA_PATH)

print("Number of recordings:", len(metadata))


# Load WavLM once
print("\nLoading WavLM Base+...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME)

model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False


# Extract embeddings
embeddings = []

start_total = time.perf_counter()

for i, row in metadata.iterrows():

    # Load recording and resample to 16 kHz
    audio, _ = librosa.load(
        row["file"],
        sr=16000,
        mono=True
    )

    # Prepare audio for WavLM
    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Frozen feature extraction
    with torch.no_grad():
        outputs = model(**inputs)

    # Temporal mean pooling
    pooled_embedding = outputs.last_hidden_state.mean(dim=1)

    # Save the 768-dimensional embedding
    embeddings.append(
        pooled_embedding.squeeze(0).cpu().numpy()
    )

    # Show progress
    print(
        f"\rProcessed {i + 1}/{len(metadata)} recordings",
        end="",
        flush=True
    )


# Combine all embeddings into one array
embeddings = np.stack(embeddings)

# Make sure output folder exists
Path("embeddings/wavlm").mkdir(
    parents=True,
    exist_ok=True
)

# Save embeddings
np.save(OUTPUT_PATH, embeddings)

total_time = time.perf_counter() - start_total


# Summary
print("\n\nExtraction complete.")
print("Embedding array shape:", embeddings.shape)
print("Total processing time:", round(total_time, 2), "seconds")
print(
    "Average time per recording:",
    round(total_time / len(metadata), 3),
    
    "seconds"
)
print("Saved to:", OUTPUT_PATH)