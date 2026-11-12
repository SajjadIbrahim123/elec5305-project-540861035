"""Evaluate noise-augmented MLPs on the original unseen-speaker test set.

Run from the project root:
    python src/evaluate_noise_augmented.py --limit 4   # trial run
    python src/evaluate_noise_augmented.py             # full experiment
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoFeatureExtractor, UniSpeechSatModel, WavLMModel

from add_noise import add_noise_at_snr, load_noise
from classifier import EmotionClassifier


SEED = 42
SAMPLE_RATE = 16000
EMOTIONS = ["neutral", "happy", "sad", "angry"]
LABEL_MAP = {emotion: index for index, emotion in enumerate(EMOTIONS)}
ENVIRONMENTS = ["DKITCHEN", "PCAFETER", "STRAFFIC"]
SNR_LEVELS = [10, 0]

MODEL_CONFIGS = [
    (
        "unispeech",
        "microsoft/unispeech-sat-base-plus",
        UniSpeechSatModel,
        "embeddings/unispeech/embeddings.npy",
        "results/unispeech_augmented_classifier.pt",
    ),
    (
        "wavlm",
        "microsoft/wavlm-base-plus",
        WavLMModel,
        "embeddings/wavlm/embeddings.npy",
        "results/wavlm_augmented_classifier.pt",
    ),
]


def predict(embeddings, classifier, mean, std):
    """Predict emotions with the classifier's saved training statistics."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    features = (embeddings - mean) / std

    with torch.inference_mode():
        logits = classifier(torch.tensor(features, dtype=torch.float32))

    return logits.argmax(dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate just the first N test recordings (trial run)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    # Never overwrite the original clean-trained model's Experiment 2 outputs.
    output_dir = Path(
        "results/noise_augmented_smoke"
        if args.limit is not None
        else "results/noise_augmented"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv("results/ravdess_metadata.csv")
    test_indices = np.flatnonzero(metadata["split"].eq("test").to_numpy())
    if args.limit is not None:
        test_indices = test_indices[:args.limit]

    test_data = metadata.iloc[test_indices]
    filenames = test_data["file"].tolist()
    y_test = test_data["emotion"].map(LABEL_MAP).to_numpy()

    if len(test_data) == 0 or pd.isna(y_test).any():
        raise ValueError("No valid test recordings or unknown emotion labels")

    print("Test recordings:", len(test_data))
    print("Test actors:", sorted(test_data["actor"].unique()))
    print("Experiment: noise-augmented classifiers")

    # Identical test audio preprocessing to the original Experiment 2.
    clean_audio = [
        librosa.load(filename, sr=SAMPLE_RATE, mono=True)[0]
        for filename in filenames
    ]
    noises = {env: load_noise(env) for env in ENVIRONMENTS}

    # IMPORTANT: identical seed, environment order, and random draws to
    # src/evaluate_noise.py, so noise segments match Experiment 2 exactly.
    rng = np.random.default_rng(SEED)
    starts = {}
    for env in ENVIRONMENTS:
        starts[env] = []
        for audio in clean_audio:
            max_start = max(0, len(noises[env]) - len(audio))
            starts[env].append(int(rng.integers(0, max_start + 1)))

    summary_rows = []
    prediction_rows = []

    def save_results(model_name, condition, predictions, environment=None, snr_db=None):
        per_class_f1 = f1_score(
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
            "environment": environment if environment is not None else "clean",
            "snr_db": snr_db,
            "num_recordings": len(y_test),
            "macro_f1": macro_f1,
            "accuracy": accuracy,
        }
        for emotion, score in zip(EMOTIONS, per_class_f1):
            row[f"f1_{emotion}"] = float(score)
        summary_rows.append(row)

        cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3])
        pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(
            output_dir / f"{model_name}_{condition}_confusion.csv"
        )

        for i, prediction in enumerate(predictions):
            prediction_rows.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "file": filenames[i],
                    "true_emotion": EMOTIONS[int(y_test[i])],
                    "predicted_emotion": EMOTIONS[int(prediction)],
                    "noise_start_sample": (
                        starts[environment][i] if environment is not None else None
                    ),
                }
            )

        print(
            f"{model_name:10s} | {condition:16s} | "
            f"Macro-F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f}"
        )

    for name, model_id, model_class, embedding_path, checkpoint_path in MODEL_CONFIGS:
        print(f"\n{'=' * 60}\nEvaluating augmented {name}\n{'=' * 60}")

        # Only the AUGMENTED MLP checkpoint differs from Experiment 2.
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        std = np.asarray(checkpoint["std"], dtype=np.float32)

        classifier = EmotionClassifier()
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.eval()

        # Clean baseline for the augmented classifier, on the SAME test actors.
        clean_embeddings = np.load(embedding_path)[test_indices]
        clean_predictions = predict(clean_embeddings, classifier, mean, std)
        save_results(name, "clean", clean_predictions)

        print("\nLoading frozen speech encoder...")
        extractor = AutoFeatureExtractor.from_pretrained(model_id)
        encoder = model_class.from_pretrained(model_id)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad = False

        for environment in ENVIRONMENTS:
            for snr_db in SNR_LEVELS:
                condition = f"{environment}_{snr_db}dB"
                embeddings = []
                print(f"\nExtracting: {condition}")

                for i, audio in enumerate(clean_audio):
                    # Same deterministic noisy audio as the original experiment.
                    noisy_audio = add_noise_at_snr(
                        audio,
                        noises[environment],
                        snr_db,
                        start_sample=starts[environment][i],
                    )
                    inputs = extractor(
                        noisy_audio,
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )
                    with torch.inference_mode():
                        outputs = encoder(**inputs)
                        pooled = outputs.last_hidden_state.mean(dim=1)

                    embeddings.append(pooled.squeeze(0).cpu().numpy())
                    print(
                        f"\rProcessed {i + 1}/{len(clean_audio)}",
                        end="",
                        flush=True,
                    )
                print()

                predictions = predict(np.stack(embeddings), classifier, mean, std)
                save_results(name, condition, predictions, environment, snr_db)

        del encoder, extractor, classifier

    summary = pd.DataFrame(summary_rows)
    clean_scores = (
        summary.loc[summary["condition"] == "clean"]
        .set_index("model")["macro_f1"]
        .to_dict()
    )
    summary["f1_change_from_clean"] = summary.apply(
        lambda row: row["macro_f1"] - clean_scores[row["model"]], axis=1
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)

    print("\nAUGMENTED CLASSIFIER RESULTS")
    print(
        summary[
            ["model", "condition", "macro_f1", "accuracy", "f1_change_from_clean"]
        ].to_string(index=False)
    )
    print("\nResults saved to:", output_dir)

    # Compare with Experiment 2 only for the FULL 112-recording evaluation.
    original_path = Path("results/noise/summary.csv")
    if args.limit is None and original_path.exists():
        original = pd.read_csv(original_path)
        if set(original["num_recordings"]) != {len(test_data)}:
            raise ValueError("Original results have a different test-set size")

        comparison = original[["model", "condition", "macro_f1"]].merge(
            summary[["model", "condition", "macro_f1"]],
            on=["model", "condition"],
            how="inner",
            suffixes=("_original", "_augmented"),
            validate="one_to_one",
        )
        if len(comparison) != len(summary):
            raise ValueError("The original and augmented conditions do not match")

        comparison["improvement"] = (
            comparison["macro_f1_augmented"] - comparison["macro_f1_original"]
        )
        comparison.to_csv(output_dir / "comparison.csv", index=False)
        print("\nORIGINAL vs AUGMENTED (positive improvement is better)")
        print(comparison.to_string(index=False, float_format=lambda n: f"{n:.4f}"))
        print("\nComparison saved to:", output_dir / "comparison.csv")


if __name__ == "__main__":
    main()
