
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoFeatureExtractor,
    UniSpeechSatModel,
    WavLMModel,
)

from add_noise import load_noise, add_noise_at_snr

SEED = 42
SAMPLE_RATE = 16000
ENVIRONMENTS = ["DKITCHEN", "PCAFETER", "STRAFFIC"]

MODELS = [
    ("unispeech", "microsoft/unispeech-sat-base-plus",
     UniSpeechSatModel),
    ("wavlm", "microsoft/wavlm-base-plus",
     WavLMModel),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    metadata = pd.read_csv("results/ravdess_metadata.csv")

    noises = {
        name: load_noise(name)
        for name in ENVIRONMENTS
    }

    for model_name, model_id, model_class in MODELS:
        print(f"\nLoading {model_name}...")

        extractor = AutoFeatureExtractor.from_pretrained(model_id)
        encoder = model_class.from_pretrained(model_id)
        encoder.eval()

        for parameter in encoder.parameters():
            parameter.requires_grad = False

        clean_embeddings = np.load(
            f"embeddings/{model_name}/embeddings.npy"
        )

        for split in ["train", "validation"]:
            indices = np.flatnonzero(
                metadata["split"].eq(split).to_numpy()
            )

            if args.limit:
                indices = indices[:args.limit]

            augmented = []

            print(f"\n{model_name} | {split}: {len(indices)} recordings")

            for count, index in enumerate(indices, 1):
                row = metadata.iloc[index]

                audio, _ = librosa.load(
                    row["file"], sr=SAMPLE_RATE, mono=True
                )

                # First embedding: original clean speech.
                augmented.append(clean_embeddings[index])

                # Two additional embeddings: 10 dB and 0 dB.
                for snr_db in [10, 0]:
                    # Fixed random selection ensures both models
                    # receive exactly the same noisy audio.
                    rng = np.random.default_rng(
                        SEED + int(index) * 2 + (snr_db == 0)
                    )

                    environment = ENVIRONMENTS[
                        int(rng.integers(len(ENVIRONMENTS)))
                    ]

                    noise = noises[environment]

                    max_start = max(0, len(noise) - len(audio))
                    start = int(rng.integers(0, max_start + 1))

                    noisy_audio = add_noise_at_snr(
                        audio, noise, snr_db,
                        start_sample=start
                    )

                    inputs = extractor(
                        noisy_audio,
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )

                    with torch.inference_mode():
                        output = encoder(**inputs)
                        embedding = (
                            output.last_hidden_state
                            .mean(dim=1)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )

                    augmented.append(embedding)

                print(
                    f"\rProcessed {count}/{len(indices)}",
                    end="",
                    flush=True,
                )

            augmented = np.stack(augmented).astype(np.float32)

            suffix = "_smoke" if args.limit else ""
            output_path = Path(
                f"embeddings/{model_name}/"
                f"augmented_{split}{suffix}.npy"
            )

            np.save(output_path, augmented)

            print(
                f"\nSaved {output_path} "
                f"with shape {augmented.shape}"
            )


if __name__ == "__main__":
    main()
