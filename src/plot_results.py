"""Creating experiment figures

"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without opening graphical windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
ORIGINAL_DIR = ROOT / "results" / "noise"
AUGMENTED_DIR = ROOT / "results" / "noise_augmented"
FIGURE_DIR = ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["neutral", "happy", "sad", "angry"]
MODELS = {"unispeech": "UniSpeech-SAT Base+", "wavlm": "WavLM Base+"}
CONDITIONS = [
    ("clean", "Clean"),
    ("DKITCHEN_10dB", "Kitchen\n10 dB"),
    ("DKITCHEN_0dB", "Kitchen\n0 dB"),
    ("PCAFETER_10dB", "Cafeteria\n10 dB"),
    ("PCAFETER_0dB", "Cafeteria\n0 dB"),
    ("STRAFFIC_10dB", "Traffic\n10 dB"),
    ("STRAFFIC_0dB", "Traffic\n0 dB"),
]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save_figure(fig, stem):
    for extension in ("png", "pdf"):
        output = FIGURE_DIR / f"{stem}.{extension}"
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        print("Saved:", output)
    plt.close(fig)


def read_summary(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    df = pd.read_csv(path)
    required = {"model", "condition", "macro_f1", *(f"f1_{e}" for e in EMOTIONS)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if df.duplicated(["model", "condition"]).any():
        raise ValueError(f"Duplicate model/condition rows in {path}")
    return df.set_index(["model", "condition"])


def get_scores(df, model, metric="macro_f1"):
    return np.asarray(
        [float(df.loc[(model, condition), metric]) for condition, _ in CONDITIONS]
    )


def draw_comparison(model, original, augmented):
    labels = [label for _, label in CONDITIONS]
    values_before = get_scores(original, model)
    values_after = get_scores(augmented, model)
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    bars_before = ax.bar(x - width / 2, values_before, width, label="Clean-trained MLP")
    bars_after = ax.bar(x + width / 2, values_after, width, label="Noise-augmented MLP")
    ax.set_title(f"{MODELS[model]}: effect of noise-augmented training", pad=15)
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x, labels)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    for bars in (bars_before, bars_after):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.014,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"macro_f1_{model}_before_after")


def draw_emotion_f1(model, original, augmented):
    condition = "PCAFETER_10dB"
    before = [float(original.loc[(model, condition), f"f1_{e}"]) for e in EMOTIONS]
    after = [float(augmented.loc[(model, condition), f"f1_{e}"]) for e in EMOTIONS]
    x = np.arange(len(EMOTIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.3, 5.1))
    ax.bar(x - width/2, before, width, label="Clean-trained MLP")
    ax.bar(x + width/2, after, width, label="Noise-augmented MLP")
    ax.set_xticks(x, [emotion.title() for emotion in EMOTIONS])
    ax.set_ylabel("Per-emotion F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{MODELS[model]}: emotion F1 under cafeteria noise (10 dB)", pad=12)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, f"per_emotion_{model}_cafeteria_10db")


def draw_confusion(model, output_label, source_dir):
    condition = "PCAFETER_10dB"
    csv_path = source_dir / f"{model}_{condition}_confusion.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing confusion matrix: {csv_path}")
    frame = pd.read_csv(csv_path, index_col=0)
    # The row labels are actual classes, and columns are predictions.
    frame = frame.loc[EMOTIONS, EMOTIONS]
    matrix = frame.to_numpy(dtype=int)
    if matrix.shape != (4, 4):
        raise ValueError(f"Unexpected confusion matrix shape: {csv_path}")

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    img = ax.imshow(matrix, vmin=0, vmax=32, aspect="equal")
    ax.set_xticks(range(4), [e.title() for e in EMOTIONS])
    ax.set_yticks(range(4), [e.title() for e in EMOTIONS])
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("Actual emotion")
    ax.set_title(
        f"{MODELS[model]} — {output_label}\nCafeteria noise, 10 dB SNR", pad=12
    )
    for row in range(4):
        for col in range(4):
            count = matrix[row, col]
            ax.text(col, row, f"{count}", ha="center", va="center",
                    color="white" if count >= 16 else "black", fontsize=12,
                    fontweight="semibold")
    fig.colorbar(img, ax=ax, label="Number of recordings", shrink=0.83)
    fig.tight_layout()
    suffix = "original" if output_label == "Clean-trained" else "augmented"
    save_figure(fig, f"confusion_{model}_cafeteria_10db_{suffix}")


def main():
    original = read_summary(ORIGINAL_DIR / "summary.csv")
    augmented = read_summary(AUGMENTED_DIR / "summary.csv")
    for model in MODELS:
        for condition, _ in CONDITIONS:
            for data in (original, augmented):
                if (model, condition) not in data.index:
                    raise ValueError(f"Missing {model} / {condition} in summary.csv")
        draw_comparison(model, original, augmented)
        draw_emotion_f1(model, original, augmented)
        draw_confusion(model, "Clean-trained", ORIGINAL_DIR)
        draw_confusion(model, "Noise-augmented", AUGMENTED_DIR)
    print("\nAll report figures are ready in results/figures/.")


if __name__ == "__main__":
    main()
