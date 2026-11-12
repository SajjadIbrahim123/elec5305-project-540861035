
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from classifier import EmotionClassifier


SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

LABEL_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
}


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


metadata = pd.read_csv("results/ravdess_metadata.csv")
labels = metadata["emotion"].map(LABEL_MAP).to_numpy()

train_labels = labels[metadata["split"].eq("train").to_numpy()]
val_labels = labels[metadata["split"].eq("validation").to_numpy()]

# Every original recording has three embeddings,
# ordered as: clean, 10 dB, 0 dB.
y_train = np.repeat(train_labels, 3)
y_val = np.repeat(val_labels, 3)


def train_augmented_model(model_name):

    set_seed()

    print(f"\n{'=' * 55}")
    print(f"Training noise-augmented {model_name} classifier")
    print(f"{'=' * 55}")

    X_train = np.load(
        f"embeddings/{model_name}/augmented_train.npy"
    )

    X_val = np.load(
        f"embeddings/{model_name}/augmented_validation.npy"
    )

    assert X_train.shape == (len(y_train), 768)
    assert X_val.shape == (len(y_val), 768)

    print("Training embeddings:", X_train.shape)
    print("Validation embeddings:", X_val.shape)

    # Compute normalisation statistics using training data only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    # Identical architecture to the original experiment.
    model = EmotionClassifier()

    class_counts = np.bincount(y_train, minlength=4)
    class_weights = len(y_train) / (4 * class_counts)

    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_score = -1.0
    best_epoch = 0

    output_path = Path(
        f"results/{model_name}_augmented_classifier.pt"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()

        with torch.no_grad():
            outputs = model(val_tensor)
            predictions = outputs.argmax(dim=1).numpy()

        # Evaluate the three validation conditions separately.
        clean_f1 = f1_score(
            y_val[0::3], predictions[0::3],
            average="macro", zero_division=0
        )

        snr10_f1 = f1_score(
            y_val[1::3], predictions[1::3],
            average="macro", zero_division=0
        )

        snr0_f1 = f1_score(
            y_val[2::3], predictions[2::3],
            average="macro", zero_division=0
        )

        # Give equal importance to clean, 10 dB, and 0 dB.
        score = (clean_f1 + snr10_f1 + snr0_f1) / 3

        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Clean: {clean_f1:.4f} | "
            f"10 dB: {snr10_f1:.4f} | "
            f"0 dB: {snr0_f1:.4f} | "
            f"Mean F1: {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "label_map": LABEL_MAP,
                    "validation_macro_f1": best_score,
                    "best_epoch": best_epoch,
                    "training_type": "noise_augmented",
                },
                output_path,
            )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best mean validation F1: {best_score:.4f}")
    print(f"Checkpoint saved: {output_path}")


train_augmented_model("unispeech")
train_augmented_model("wavlm")

print("\nBoth augmented classifiers trained successfully.")
