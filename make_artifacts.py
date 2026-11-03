from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.inference import ParityPredictor


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for tag, color in [("plain", "#8a8f98"), ("augmented", "#245b3c")]:
        rows = list(csv.DictReader((ART / f"history_{tag}.csv").open()))
        epoch = [int(r["epoch"]) for r in rows]
        axes[0].plot(epoch, [float(r["train_loss"]) for r in rows], label=tag, color=color)
        axes[1].plot(epoch, [float(r["val_f1"]) for r in rows], label=tag, color=color)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1.05))
    for ax in axes: ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(ART / "training_ablation_curves.png", dpi=200); plt.close(fig)


def comparisons():
    baseline = json.loads((ART / "baseline_metrics.json").read_text())
    plain = json.loads((ART / "metrics_plain.json").read_text())
    aug = json.loads((ART / "metrics_augmented.json").read_text())
    legacy = {"accuracy": .9666666667, "precision": .9375, "recall": 1.0, "f1": .9677419355, "latency_ms": None}
    rows = [
        ("Logistic Regression", baseline),
        ("Legacy CNN", legacy),
        ("SE-ResNet no augmentation", plain),
        ("SE-ResNet domain augmentation", aug),
    ]
    with (ART / "comparison.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "Latency_ms"])
        for name, r in rows: w.writerow([name, r["accuracy"], r["precision"], r["recall"], r["f1"], r.get("latency_ms", "")])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [x[0] for x in rows]
    x = np.arange(len(names)); width = .36
    ax.bar(x - width/2, [x[1]["accuracy"] for x in rows], width, label="Accuracy", color="#8ba99a")
    ax.bar(x + width/2, [x[1]["f1"] for x in rows], width, label="F1", color="#245b3c")
    ax.set_ylim(.6, 1.01); ax.set_xticks(x, names, rotation=12, ha="right"); ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
    ax.set_title("Test-set model comparison (n=30)")
    fig.tight_layout(); fig.savefig(ART / "model_comparison.png", dpi=200); plt.close(fig)


def confusion_and_gradcam():
    pred_data = json.loads((ART / "predictions_augmented.json").read_text())
    cm = confusion_matrix(pred_data["labels"], pred_data["predictions"])
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ConfusionMatrixDisplay(cm, display_labels=["even", "odd"]).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("SE-ResNet confusion matrix")
    fig.tight_layout(); fig.savefig(ART / "se_resnet_confusion_matrix.png", dpi=200); plt.close(fig)

    predictor = ParityPredictor(ART / "se_resnet_augmented.pt")
    candidates = [
        sorted((ROOT / "data/test/even").glob("*.png"))[0],
        sorted((ROOT / "data/test/odd").glob("*.png"))[0],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for row, path in enumerate(candidates):
        image = Image.open(path).convert("RGB")
        result = predictor.predict(image)
        axes[row, 0].imshow(image, cmap="gray"); axes[row, 0].set_title(f"Input: {path.parent.name}")
        axes[row, 1].imshow(result["overlay"]); axes[row, 1].set_title(f"Grad-CAM: {result['label']} ({result['confidence']:.0%})")
        axes[row, 0].axis("off"); axes[row, 1].axis("off")
    fig.tight_layout(); fig.savefig(ART / "gradcam_examples.png", dpi=200); plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off")
    labels = ["64x64\ngrayscale", "3x3 stem\n24 channels", "SE residual\n24", "SE residual\n48, stride 2", "SE residual\n96, stride 2", "global pool\n+ dropout", "even / odd"]
    xs = np.linspace(.05, .95, len(labels))
    colors = ["#e9efea", "#d6e4da", "#bfd5c5", "#9fc2aa", "#79a98b", "#d6e4da", "#245b3c"]
    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        ax.text(x, .5, label, ha="center", va="center", fontsize=9, color="white" if i == 6 else "#142019",
                bbox=dict(boxstyle="round,pad=.55", facecolor=color, edgecolor="#9aa69d"))
        if i < len(labels)-1: ax.annotate("", (xs[i+1]-.065, .5), (x+.065, .5), arrowprops=dict(arrowstyle="->", color="#637067"))
    ax.set_title("Compact SE-ResNet pipeline", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(ART / "architecture_diagram.png", dpi=200, transparent=False); plt.close(fig)


if __name__ == "__main__":
    training_curves(); comparisons(); confusion_and_gradcam(); architecture()
    print("Wrote report artifacts to", ART)
