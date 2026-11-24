"""Training script for 5-class PyTorch manuscript script classifier.

Classes: Kannada, Tamil, Telugu, Arabic, Latin.
Saves model checkpoint to models/script_classifier/best_script_classifier.pth
"""

import argparse
import os
import time
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from src.script_classifier.dataset import get_train_val_test_dataloaders
from src.script_classifier.model import CLASS_NAMES, ScriptClassifierCNN
from src.utils import get_logger

LOGGER = get_logger("train_script_classifier")


def train_classifier(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    samples_per_class: int = 150,
    save_dir: str = "models/script_classifier",
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[ScriptClassifierCNN, Dict[str, float]]:
    """Train the 5-class script classifier CNN and save the best checkpoint."""
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    device_obj = torch.device(device)

    LOGGER.info("Generating dataset: %d samples/class across %d scripts...", samples_per_class, len(CLASS_NAMES))
    train_loader, val_loader, test_loader, splits = get_train_val_test_dataloaders(
        batch_size=batch_size,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    LOGGER.info("Splits: Train=%d, Val=%d, Test=%d", splits["train"], splits["val"], splits["test"])

    model = ScriptClassifierCNN(num_classes=len(CLASS_NAMES), in_channels=1).to(device_obj)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(save_dir, "best_script_classifier.pth")
    training_history = []

    LOGGER.info("Starting CNN training for %d epochs on %s...", epochs, device)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_obj), targets.to(device_obj)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_obj), targets.to(device_obj)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)
        scheduler.step(epoch_val_acc)

        LOGGER.info(
            "Epoch [%02d/%02d] - Train Loss: %.4f | Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
            epoch, epochs, epoch_train_loss, epoch_train_acc * 100, epoch_val_loss, epoch_val_acc * 100,
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4),
        })

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "classes": CLASS_NAMES,
                "history": training_history,
            }, best_checkpoint_path)

    elapsed = time.time() - start_time
    LOGGER.info("Training complete in %.2fs. Best Val Accuracy: %.2f%%", elapsed, best_val_acc * 100)
    LOGGER.info("Saved best model to: %s", best_checkpoint_path)

    metrics = {
        "best_val_accuracy": round(best_val_acc, 4),
        "total_time_seconds": round(elapsed, 2),
        "checkpoint_path": best_checkpoint_path,
    }
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 5-class manuscript script classifier")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples-per-class", type=int, default=150, help="Samples generated per class")
    parser.add_argument("--save-dir", type=str, default="models/script_classifier", help="Save directory")
    args = parser.parse_args()

    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_class=args.samples_per_class,
        save_dir=args.save_dir,
    )
