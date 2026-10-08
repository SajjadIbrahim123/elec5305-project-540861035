from pathlib import Path
import pandas as pd

# Location of the RAVDESS dataset
DATASET_PATH = Path("data/ravdess")

# Emotions used in this project
EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

records = []

# Search through every WAV file in every Actor folder
for audio_path in sorted(DATASET_PATH.glob("Actor_*/*.wav")):

    # Example:
    # 03-01-05-01-01-01-01.wav
    parts = audio_path.stem.split("-")

    emotion_code = parts[2]
    actor_id = int(parts[6])

    # Keep only neutral, happy, sad and angry
    if emotion_code not in EMOTION_MAP:
        continue

    records.append({
        "file": str(audio_path),
        "actor": actor_id,
        "emotion": EMOTION_MAP[emotion_code],
    })

# Create a table containing the dataset information
metadata = pd.DataFrame(records)

print(metadata.head(10))

print("\nNumber of selected recordings:", len(metadata))
print("Number of actors:", metadata["actor"].nunique())

print("\nRecordings per emotion:")
print(metadata["emotion"].value_counts())

print("\nRecordings per actor:")
print(metadata["actor"].value_counts().sort_index())

# Actor-independent train/validation/test split
train_actors = list(range(1, 17))
val_actors = list(range(17, 21))
test_actors = list(range(21, 25))

def assign_split(actor):
    if actor in train_actors:
        return "train"
    elif actor in val_actors:
        return "validation"
    else:
        return "test"

metadata["split"] = metadata["actor"].apply(assign_split)

print("\nDataset split:")
print(metadata["split"].value_counts())

print("\nActors in each split:")
for split in ["train", "validation", "test"]:
    actors = sorted(metadata[metadata["split"] == split]["actor"].unique())
    print(f"{split}: {actors}")

# Save metadata so both models use exactly the same dataset and splits
metadata.to_csv("results/ravdess_metadata.csv", index=False)

print("\nSaved metadata to results/ravdess_metadata.csv")