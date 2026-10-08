import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from classifier import EmotionClassifier


# --------------------------------------------------
# Settings
# --------------------------------------------------

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


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

metadata = pd.read_csv("results/ravdess_metadata.csv")
labels = metadata["emotion"].map(LABEL_MAP).to_numpy()

train_mask = metadata["split"].eq("train").to_numpy()
val_mask = metadata["split"].eq("validation").to_numpy()


# --------------------------------------------------
# Train one classifier
# --------------------------------------------------

def train_model(model_name, embedding_path, output_path):

    print(f"\n{'=' * 50}")
    print(f"Training classifier using {model_name}")
    print(f"{'=' * 50}")

    embeddings = np.load(embedding_path)

    # Actor-independent split
    X_train = embeddings[train_mask]
    y_train = labels[train_mask]

    X_val = embeddings[val_mask]
    y_val = labels[val_mask]

    print("Training recordings:", len(X_train))
    print("Validation recordings:", len(X_val))

    # Standardise using TRAINING data only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Training batches
    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Create identical MLP
    model = EmotionClassifier()

    # Account for the smaller neutral class
    class_counts = np.bincount(y_train.numpy(), minlength=4)
    class_weights = len(y_train) / (4 * class_counts)

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_f1 = -1.0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

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

        # Validation
        model.eval()

        with torch.no_grad():

            val_outputs = model(X_val)

            val_predictions = torch.argmax(
                val_outputs,
                dim=1
            ).numpy()

        val_f1 = f1_score(
            y_val,
            val_predictions,
            average="macro"
        )

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Macro-F1: {val_f1:.4f}"
        )

        # Save the best validation model
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "label_map": LABEL_MAP,
                    "validation_macro_f1": best_val_f1,
                },
                output_path
            )

    print(
        f"\nBest {model_name} validation Macro-F1: "
        f"{best_val_f1:.4f}"
    )


# --------------------------------------------------
# Train both representation systems
# --------------------------------------------------

train_model(
    "UniSpeech-SAT",
    "embeddings/unispeech/embeddings.npy",
    "results/unispeech_classifier.pt"
)

# Reset seed so WavLM receives the same classifier
# initialisation and training randomness
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_model(
    "WavLM",
    "embeddings/wavlm/embeddings.npy",
    "results/wavlm_classifier.pt"

)
