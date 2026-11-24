"""Evaluation script for 5-class script classifier CNN.

Computes accuracy, precision, recall, F1-score, and generates
the confusion matrix image saved to docs/script_classifier_confusion_matrix.png.
"""

import json
import os
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.script_classifier.dataset import get_train_val_test_dataloaders
from src.script_classifier.model import CLASS_NAMES, ScriptClassifierCNN
from src.utils import get_logger

LOGGER = get_logger("evaluate_script_classifier")


def evaluate_model(
    checkpoint_path: str = "models/script_classifier/best_script_classifier.pth",
    output_dir: str = "docs",
    batch_size: int = 32,
    samples_per_class: int = 150,
    seed: int = 42,
) -> Dict[str, any]:
    """Evaluate trained script classifier on test set and plot confusion matrix."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Run training first.")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Load test split (same seed ensures exact split reproducibility)
    _, _, test_loader, splits = get_train_val_test_dataloaders(
        batch_size=batch_size,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ScriptClassifierCNN(num_classes=5, in_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    total_samples = len(y_true)
    correct = int((y_true == y_pred).sum())
    overall_accuracy = float(correct / max(1, total_samples))

    # Compute 5x5 confusion matrix
    n_classes = len(CLASS_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for i, cname in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_metrics[cname] = {
            "samples": int(cm[i, :].sum()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="5-Class Manuscript Script Classifier - Confusion Matrix\n"
              "(Expected structural proximity between Kannada and Telugu scripts)",
        ylabel="True Script",
        xlabel="Predicted Script",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Add numeric labels to cells
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.tight_layout()
    cm_path = os.path.join(output_dir, "script_classifier_confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    results = {
        "overall_accuracy": round(overall_accuracy, 4),
        "total_test_samples": total_samples,
        "correct_predictions": correct,
        "classes": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix_image": cm_path,
    }

    metrics_json_path = os.path.join(output_dir, "script_classifier_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    LOGGER.info("Evaluation complete! Overall Test Accuracy: %.2f%%", overall_accuracy * 100)
    LOGGER.info("Confusion matrix plot saved to: %s", cm_path)
    return results


if __name__ == "__main__":
    evaluate_model()
