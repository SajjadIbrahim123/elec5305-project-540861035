
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoFeatureExtractor,
    UniSpeechSatModel,
    WavLMModel,
)

from add_noise import load_noise, add_noise_at_snr
from classifier import EmotionClassifier


SEED = 42
SAMPLE_RATE = 16000
EMOTIONS = ["neutral", "happy", "sad", "angry"]
LABEL_MAP = {name: i for i, name in enumerate(EMOTIONS)}
ENVIRONMENTS = ["DKITCHEN", "PCAFETER", "STRAFFIC"]
SNR_LEVELS = [10, 0]

MODEL_CONFIGS = [
    (
        "unispeech",
        "microsoft/unispeech-sat-base-plus",
        UniSpeechSatModel,
        "embeddings/unispeech/embeddings.npy",
        "results/unispeech_classifier.pt",
    ),
    (
        "wavlm",
        "microsoft/wavlm-base-plus",
        WavLMModel,
        "embeddings/wavlm/embeddings.npy",
        "results/wavlm_classifier.pt",
    ),
]


def predict(embeddings, classifier, mean, std):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    features = (embeddings - mean) / std

    with torch.inference_mode():
        logits = classifier(
            torch.tensor(features, dtype=torch.float32)
        )

    return logits.argmax(dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of test recordings for a trial run",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    output_dir = Path(
        "results/noise_smoke" if args.limit else "results/noise"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv("results/ravdess_metadata.csv")

    # Preserve the original metadata order so that embedding
    # rows and recording labels remain aligned.
    test_indices = np.flatnonzero(
        metadata["split"].eq("test").to_numpy()
    )

    if args.limit:
        test_indices = test_indices[:args.limit]

    test_data = metadata.iloc[test_indices]
    filenames = test_data["file"].tolist()
    y_test = test_data["emotion"].map(LABEL_MAP).to_numpy()

    print("Test recordings:", len(test_data))
    print("Test actors:", sorted(test_data["actor"].unique()))

    # Load clean test audio once, at the original 16 kHz
    # processing rate used in Experiment 1.
    clean_audio = []
    for filename in filenames:
        audio, _ = librosa.load(
            filename, sr=SAMPLE_RATE, mono=True
        )
        clean_audio.append(audio)

    # Load environmental noise once.
    noises = {
        env: load_noise(env)
        for env in ENVIRONMENTS
    }

    # Choose deterministic noise segments for every recording.
    # Both models and both SNR values use the SAME segments.
    rng = np.random.default_rng(SEED)
    starts = {}

    for env in ENVIRONMENTS:
        starts[env] = []
        for audio in clean_audio:
            max_start = max(0, len(noises[env]) - len(audio))
            start = int(rng.integers(0, max_start + 1))
            starts[env].append(start)

    summary_rows = []
    prediction_rows = []

    def save_results(model_name, condition, predictions, environment=None,
                     snr_db=None):
        scores = f1_score(
            y_test,
            predictions,
            labels=[0, 1, 2, 3],
            average=None,
            zero_division=0,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )

        accuracy = accuracy_score(y_test, predictions)

        row = {
            "model": model_name,
            "condition": condition,
            "environment": environment or "clean",
            "snr_db": snr_db,
            "num_recordings": len(y_test),
            "macro_f1": macro_f1,
            "accuracy": accuracy,
        }

        for emotion, score in zip(EMOTIONS, scores):
            row[f"f1_{emotion}"] = score

        summary_rows.append(row)

        cm = confusion_matrix(
            y_test, predictions, labels=[0, 1, 2, 3]
        )

        cm_path = output_dir / (
            f"{model_name}_{condition}_confusion.csv"
        )

        pd.DataFrame(
            cm, index=EMOTIONS, columns=EMOTIONS
        ).to_csv(cm_path)

        for i, prediction in enumerate(predictions):
            prediction_rows.append({
                "model": model_name,
                "condition": condition,
                "file": filenames[i],
                "true_emotion": EMOTIONS[y_test[i]],
                "predicted_emotion": EMOTIONS[prediction],
                "noise_start_sample": (
                    starts[environment][i]
                    if environment is not None else None
                ),
            })

        print(
            f"{model_name:10s} | {condition:16s} | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

    for name, model_id, model_class, embedding_path, checkpoint_path in MODEL_CONFIGS:
        print(f"\n{'=' * 60}")
        print("Evaluating:", name)
        print(f"{'=' * 60}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        std = np.asarray(checkpoint["std"], dtype=np.float32)

        classifier = EmotionClassifier()
        classifier.load_state_dict(
            checkpoint["model_state_dict"]
        )
        classifier.eval()

        # Reproduce the original clean baseline.
        clean_embeddings = np.load(embedding_path)[test_indices]
        clean_predictions = predict(
            clean_embeddings, classifier, mean, std
        )
        save_results(name, "clean", clean_predictions)

        print("\nLoading frozen speech encoder...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id
        )
        encoder = model_class.from_pretrained(model_id)
        encoder.eval()

        for parameter in encoder.parameters():
            parameter.requires_grad = False

        # Experiment 2: three noise environments, two SNR levels.
        for environment in ENVIRONMENTS:
            for snr_db in SNR_LEVELS:
                condition = f"{environment}_{snr_db}dB"
                embeddings = []

                print(f"\nExtracting: {condition}")

                for i, audio in enumerate(clean_audio):
                    noisy_audio = add_noise_at_snr(
                        audio,
                        noises[environment],
                        snr_db,
                        start_sample=starts[environment][i],
                    )

                    inputs = feature_extractor(
                        noisy_audio,
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )

                    with torch.inference_mode():
                        outputs = encoder(**inputs)
                        pooled = outputs.last_hidden_state.mean(
                            dim=1
                        )

                    embeddings.append(
                        pooled.squeeze(0).cpu().numpy()
                    )

                    print(
                        f"\rProcessed {i + 1}/{len(clean_audio)}",
                        end="",
                        flush=True,
                    )

                print()

                predictions = predict(
                    np.stack(embeddings),
                    classifier,
                    mean,
                    std,
                )

                save_results(
                    name,
                    condition,
                    predictions,
                    environment,
                    snr_db,
                )

        del encoder, feature_extractor, classifier

    summary = pd.DataFrame(summary_rows)

    # Compare each noisy result against the SAME model's clean result.
    clean_scores = (
        summary[summary["condition"] == "clean"]
        .set_index("model")["macro_f1"]
        .to_dict()
    )

    summary["f1_change_from_clean"] = summary.apply(
        lambda row: row["macro_f1"] - clean_scores[row["model"]],
        axis=1,
    )

    summary.to_csv(
        output_dir / "summary.csv", index=False
    )

    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "predictions.csv", index=False
    )

    print("\nEXPERIMENT COMPLETE")
    print(summary[
        ["model", "condition", "macro_f1",
         "accuracy", "f1_change_from_clean"]
    ].to_string(index=False))

    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    main()
