import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report
)

from classifier import EmotionClassifier


LABEL_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
}

CLASS_NAMES = ["neutral", "happy", "sad", "angry"]


# Load metadata
metadata = pd.read_csv("results/ravdess_metadata.csv")

labels = metadata["emotion"].map(LABEL_MAP).to_numpy()

# Use ONLY the untouched test actors
test_mask = metadata["split"].eq("test").to_numpy()

y_test = labels[test_mask]

print("Number of test recordings:", len(y_test))
print(
    "Test actors:",
    sorted(metadata.loc[test_mask, "actor"].unique())
)


def evaluate_model(model_name, embedding_path, checkpoint_path):

    print(f"\n{'=' * 50}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 50}")

    # Load embeddings
    embeddings = np.load(embedding_path)
    X_test = embeddings[test_mask]

    # Load the best validation checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    # Use training-set statistics saved during training
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    X_test = (X_test - mean) / std

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # Recreate classifier and load learned weights
    model = EmotionClassifier()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Make predictions
    with torch.no_grad():

        outputs = model(X_test)

        predictions = torch.argmax(
            outputs,
            dim=1
        ).numpy()

    # Overall Macro-F1
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(f"\nClean Test Macro-F1: {macro_f1:.4f}")

    # Per-emotion results
    print("\nPer-emotion results:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3]
    )

    print("Confusion matrix:")
    print(cm)

    return macro_f1


# Evaluate UniSpeech-SAT
unispeech_f1 = evaluate_model(
    "UniSpeech-SAT",
    "embeddings/unispeech/embeddings.npy",
    "results/unispeech_classifier.pt"
)


# Evaluate WavLM
wavlm_f1 = evaluate_model(
    "WavLM",
    "embeddings/wavlm/embeddings.npy",
    "results/wavlm_classifier.pt"
)


# Final comparison
print("\n" + "=" * 50)
print("CLEAN UNSEEN-SPEAKER RESULTS")
print("=" * 50)

print(f"UniSpeech-SAT Macro-F1: {unispeech_f1:.4f}")
print(f"WavLM Macro-F1:         {wavlm_f1:.4f}")
