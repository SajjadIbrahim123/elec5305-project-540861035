"""Creating tables from the saved results.

"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ORIGINAL = Path("results/noise/summary.csv")
AUGMENTED = Path("results/noise_augmented/summary.csv")
OUTPUT = Path("results/tables")

CONDITIONS = [
    ("clean", "Clean"),
    ("DKITCHEN_10dB", "Kitchen (10 dB)"),
    ("DKITCHEN_0dB", "Kitchen (0 dB)"),
    ("PCAFETER_10dB", "Cafeteria (10 dB)"),
    ("PCAFETER_0dB", "Cafeteria (0 dB)"),
    ("STRAFFIC_10dB", "Traffic (10 dB)"),
    ("STRAFFIC_0dB", "Traffic (0 dB)"),
]
EMOTIONS = ["neutral", "happy", "sad", "angry"]
MODELS = ["unispeech", "wavlm"]


def use_research_font():
    """Select actual Times New Roman if installed (normally present on macOS)."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        family = "Times New Roman"
    else:
        family = next(
            (name for name in ("Tinos", "Liberation Serif", "DejaVu Serif")
             if name in available),
            "DejaVu Serif",
        )
        warnings.warn(
            f"Times New Roman is not installed; using {family}. "
            "Install Times New Roman to use the exact requested font.",
            stacklevel=2,
        )
    plt.rcParams.update({
        "font.family": family,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    print("Table font:", family)
    return family


def read_results(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the evaluation script first.")
    df = pd.read_csv(path)
    required = {
        "model", "condition", "macro_f1", "accuracy", "num_recordings",
        *(f"f1_{emotion}" for emotion in EMOTIONS),
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns from {path}: {sorted(missing)}")
    if df.duplicated(["model", "condition"]).any():
        raise ValueError(f"Duplicate model/condition combinations in {path}")
    df = df.set_index(["model", "condition"])
    for model in MODELS:
        for condition, _ in CONDITIONS:
            key = (model, condition)
            if key not in df.index:
                raise ValueError(f"Missing {key} in {path}")
            row = df.loc[key]
            if int(row["num_recordings"]) != 112:
                raise ValueError(f"Expected 112 test recordings for {key} in {path}")
            for metric in ["macro_f1", "accuracy", *(f"f1_{e}" for e in EMOTIONS)]:
                value = float(row[metric])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid value of {metric} for {key}: {value}")
    return df


def metric(df, model, condition, field):
    return float(df.loc[(model, condition), field])


def render_table(data, filename, number, title, note, headings, *, signed=False):
    """Draw a clean academic table: no colored cells and no vertical rules."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT / f"{filename}.csv", index=False, float_format="%.6f")

    row_count = len(data)
    column_count = len(data.columns)
    fig_width = 9.0
    fig_height = 2.12 + 0.39 * row_count
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    left, right = 0.45, fig_width - 0.45
    usable_width = right - left
    if column_count == 5:
        proportions = [0.260, 0.185, 0.185, 0.185, 0.185]
    else:
        proportions = [0.38, 0.31, 0.31]
    bounds = left + usable_width * np.r_[0, np.cumsum(proportions)]

    ax.text(left, fig_height - 0.30, f"Table {number}",
            fontsize=11.5, fontweight="bold", ha="left", va="center")
    ax.text(left, fig_height - 0.59, title,
            fontsize=11.5, fontstyle="italic", ha="left", va="center")

    top_line = fig_height - 0.98
    header_lower = top_line - 0.56
    ax.hlines(top_line, left, right, color="black", linewidth=1.15)
    ax.hlines(header_lower, left, right, color="black", linewidth=0.72)

    for j, heading in enumerate(headings):
        x = (bounds[j] + bounds[j + 1]) / 2
        ax.text(x, (top_line + header_lower) / 2, heading,
                fontsize=9.8 if column_count == 5 else 10.7,
                ha="center", va="center", linespacing=1.2)

    for i, values in enumerate(data.itertuples(index=False, name=None)):
        y = header_lower - 0.39 * (i + 0.5)
        ax.text(bounds[0] + 0.09, y, str(values[0]),
                ha="left", va="center", fontsize=11)
        for j, value in enumerate(values[1:], start=1):
            if isinstance(value, (float, np.floating)):
                formatted = f"{value:+.4f}" if signed else f"{value:.4f}"
            else:
                formatted = str(value)
            ax.text(bounds[j + 1] - 0.14, y, formatted,
                    ha="right", va="center", fontsize=11)

    bottom_line = header_lower - 0.39 * row_count
    ax.hlines(bottom_line, left, right, color="black", linewidth=1.15)
    ax.text(left, bottom_line - 0.25, f"Note. {note}",
            fontsize=9.3, ha="left", va="center")

    for extension in ["png", "pdf"]:
        path = OUTPUT / f"{filename}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print("Saved:", path)
    plt.close(fig)
    print("Saved:", OUTPUT / f"{filename}.csv")


def main():
    use_research_font()
    original = read_results(ORIGINAL)
    augmented = read_results(AUGMENTED)

    columns = [
        "Condition", "UniSpeech-SAT original", "UniSpeech-SAT augmented",
        "WavLM original", "WavLM augmented",
    ]
    headings = [
        "Audio condition", "UniSpeech-SAT\nClean-trained",
        "UniSpeech-SAT\nNoise-augmented", "WavLM\nClean-trained",
        "WavLM\nNoise-augmented",
    ]

    def build_comparison(field):
        return pd.DataFrame([
            [label,
             metric(original, "unispeech", code, field),
             metric(augmented, "unispeech", code, field),
             metric(original, "wavlm", code, field),
             metric(augmented, "wavlm", code, field)]
            for code, label in CONDITIONS
        ], columns=columns)

    general_note = "Results use 112 recordings from unseen RAVDESS actors 21-24; higher scores are better."
    render_table(
        build_comparison("macro_f1"), "macro_f1_comparison", 1,
        "Macro-F1 across clean and noisy test conditions",
        general_note, headings,
    )

    gains = pd.DataFrame([
        [label,
         metric(augmented, "unispeech", code, "macro_f1")
         - metric(original, "unispeech", code, "macro_f1"),
         metric(augmented, "wavlm", code, "macro_f1")
         - metric(original, "wavlm", code, "macro_f1")]
        for code, label in CONDITIONS
    ], columns=["Condition", "UniSpeech-SAT change", "WavLM change"])
    render_table(
        gains, "macro_f1_improvement", 2,
        "Change in macro-F1 after noise-augmented training",
        "Change = noise-augmented minus clean-trained macro-F1; positive values indicate gains.",
        ["Audio condition", "UniSpeech-SAT\nChange in macro-F1", "WavLM\nChange in macro-F1"],
        signed=True,
    )

    render_table(
        build_comparison("accuracy"), "accuracy_comparison", 3,
        "Classification accuracy across test conditions",
        general_note, headings,
    )

    cafeteria = "PCAFETER_10dB"
    per_emotion = pd.DataFrame([
        [emotion.capitalize(),
         metric(original, "unispeech", cafeteria, f"f1_{emotion}"),
         metric(augmented, "unispeech", cafeteria, f"f1_{emotion}"),
         metric(original, "wavlm", cafeteria, f"f1_{emotion}"),
         metric(augmented, "wavlm", cafeteria, f"f1_{emotion}")]
        for emotion in EMOTIONS
    ], columns=["Emotion", *columns[1:]])
    render_table(
        per_emotion, "cafeteria_per_emotion_f1", 4,
        "Per-emotion F1 with cafeteria noise at 10 dB SNR",
        "Results use the same 112 unseen-speaker recordings; higher scores are better.",
        ["Emotion", *headings[1:]],
    )

    print("\nFinished. All four tables are in results/tables/.")


if __name__ == "__main__":
    main()
