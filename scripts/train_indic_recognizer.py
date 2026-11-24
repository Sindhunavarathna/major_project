"""Training script for Shared Indic (Kannada, Tamil, Telugu) CRNN Handwriting Recognizer.

Supports:
- Free Google Colab GPU (CUDA) and CPU
- Connectionist Temporal Classification (CTCLoss)
- Unified character vocabulary covering all 3 Dravidian scripts
- Zero-leakage writer-separated train/val/test splits
- Per-epoch logging of Train Loss, Val Loss, CER and WER per language
- Checkpointing best model and resuming with --resume
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indic_ocr.charset import UnifiedIndicCharset
from src.indic_ocr.dataset import (
    IndicCRNNDataset,
    IndicSample,
    generate_seed_indic_samples,
    indic_collate_fn,
    load_iiit_indic_samples,
    load_team_collected_samples,
    split_samples_by_writer,
)
from src.indic_ocr.metrics import compute_corpus_metrics
from src.indic_ocr.model import IndicCRNN
from src.utils import get_logger

LOGGER = get_logger("train_indic_recognizer")


def train_indic_crnn(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 5e-4,
    save_dir: str = "models/indic_crnn",
    resume_path: str = "",
    device_name: str = "auto",
    seed: int = 42,
) -> Tuple[IndicCRNN, UnifiedIndicCharset, Dict[str, float]]:
    """Train the shared Indic CRNN recognizer across Kannada, Tamil, and Telugu."""
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    LOGGER.info("Executing on device: %s", device)

    # 1. Collect all available Indic handwriting data
    samples: List[IndicSample] = []
    # Try IIIT-INDIC-HW-WORDS
    iiit_samples = load_iiit_indic_samples("data")
    if iiit_samples:
        LOGGER.info("Loaded %d samples from IIIT-INDIC-HW-WORDS", len(iiit_samples))
        samples.extend(iiit_samples)

    # Try team-collected samples
    team_samples = load_team_collected_samples("data/collected")
    if team_samples:
        LOGGER.info("Loaded %d samples from team collected directory", len(team_samples))
        samples.extend(team_samples)

    # If dataset has fewer than 180 samples, complement with seed generator
    if len(samples) < 180:
        LOGGER.info("Dataset size (%d) below target; synthesizing verified multi-writer seed samples...", len(samples))
        seed_samples = generate_seed_indic_samples(writers_per_lang=8, words_per_writer=15, seed=seed)
        samples.extend(seed_samples)

    # 2. Split strictly by writer with zero leakage
    train_s, val_s, test_s, split_counts = split_samples_by_writer(
        samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )

    print("\n" + "=" * 65)
    print("  SHARED INDIC DATASET: WRITER-SEPARATED SPLITS (ZERO LEAKAGE)")
    print("=" * 65)
    for lang, counts in split_counts.items():
        print(f"[{lang.upper()}] Total Writers: {counts['total_writers']} (Train: {counts['train_writers']}, Val: {counts['val_writers']}, Test: {counts['test_writers']})")
        print(f"         Samples -> Train: {counts['train_samples']} | Val: {counts['val_samples']} | Test: {counts['test_samples']}")
    print(f"TOTAL DATASET: Train={len(train_s)} | Val={len(val_s)} | Test={len(test_s)} samples")
    print("=" * 65 + "\n")

    # 3. Initialize and save unified character vocabulary
    charset = UnifiedIndicCharset()
    vocab_file = os.path.join(save_dir, "vocab.json")
    charset.save(vocab_file)
    LOGGER.info("Unified vocabulary saved to: %s (Total classes: %d)", vocab_file, len(charset))

    # 4. Create PyTorch DataLoaders
    train_ds = IndicCRNNDataset(train_s, charset, augment=True)
    val_ds = IndicCRNNDataset(val_s, charset, augment=False)
    test_ds = IndicCRNNDataset(test_s, charset, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=indic_collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=indic_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=indic_collate_fn)

    # 5. Build CRNN model
    model = IndicCRNN(num_classes=len(charset)).to(device)
    criterion = nn.CTCLoss(blank=charset.BLANK_INDEX, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    start_epoch = 1
    best_val_cer = float("inf")
    best_checkpoint_path = os.path.join(save_dir, "best_indic_crnn.pth")
    history = []

    # Optional resume
    if resume_path and os.path.exists(resume_path):
        LOGGER.info("Resuming training from checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_cer = ckpt.get("val_cer", float("inf"))

    LOGGER.info("Starting Indic CRNN training for %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for images, targets_concat, input_lengths, target_lengths, _, _ in train_loader:
            images = images.to(device)
            targets_concat = targets_concat.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(images)  # (T, B, num_classes)
            loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)

        # Validation Phase: compute Loss, CER and WER per language
        model.eval()
        val_loss = 0.0
        val_batches = 0

        lang_refs: Dict[str, List[str]] = {"Kannada": [], "Tamil": [], "Telugu": []}
        lang_hyps: Dict[str, List[str]] = {"Kannada": [], "Tamil": [], "Telugu": []}

        with torch.no_grad():
            for images, targets_concat, input_lengths, target_lengths, raw_texts, languages in val_loader:
                images = images.to(device)
                targets_concat = targets_concat.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                log_probs = model(images)  # (T, B, num_classes)
                loss = criterion(log_probs, targets_concat, input_lengths, target_lengths)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1

                # Greedy CTC decoding for validation samples
                # log_probs is (T, B, C) -> transpose to (B, T, C)
                log_probs_b = log_probs.permute(1, 0, 2)
                for b_idx in range(images.size(0)):
                    sample_lang = languages[b_idx]
                    ref_text = raw_texts[b_idx]
                    hyp_text = charset.decode_greedy(log_probs_b[b_idx])
                    if sample_lang in lang_refs:
                        lang_refs[sample_lang].append(ref_text)
                        lang_hyps[sample_lang].append(hyp_text)

        avg_val_loss = val_loss / max(1, val_batches)
        scheduler.step(avg_val_loss)

        # Compute CER and WER per language
        lang_cer = {}
        lang_wer = {}
        all_refs = []
        all_hyps = []
        for l in ["Kannada", "Tamil", "Telugu"]:
            c_cer, c_wer = compute_corpus_metrics(lang_refs[l], lang_hyps[l])
            lang_cer[l] = c_cer
            lang_wer[l] = c_wer
            all_refs.extend(lang_refs[l])
            all_hyps.extend(lang_hyps[l])

        overall_cer, overall_wer = compute_corpus_metrics(all_refs, all_hyps)

        LOGGER.info(
            "Epoch [%02d/%02d] - Loss: (Train=%.4f, Val=%.4f) | CER: Kn=%.2f%%, Ta=%.2f%%, Te=%.2f%% | Overall CER=%.2f%%, WER=%.2f%%",
            epoch, epochs, avg_train_loss, avg_val_loss,
            lang_cer["Kannada"] * 100, lang_cer["Tamil"] * 100, lang_cer["Telugu"] * 100,
            overall_cer * 100, overall_wer * 100,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "overall_cer": overall_cer,
            "overall_wer": overall_wer,
            "per_lang_cer": lang_cer,
            "per_lang_wer": lang_wer,
        })

        if overall_cer < best_val_cer or epoch == 1:
            best_val_cer = overall_cer
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab": charset._chars,
                "val_cer": best_val_cer,
                "val_wer": overall_wer,
                "per_lang_cer": lang_cer,
                "per_lang_wer": lang_wer,
                "split_counts": split_counts,
                "history": history,
            }, best_checkpoint_path)

    elapsed = time.time() - start_time
    LOGGER.info("Training complete in %.2fs. Saved best model to: %s", elapsed, best_checkpoint_path)

    metrics = {
        "best_val_cer": round(best_val_cer, 4),
        "total_time_seconds": round(elapsed, 2),
        "checkpoint_path": best_checkpoint_path,
        "history": history,
    }
    return model, charset, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Shared Indic CRNN Handwriting Recognizer")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="models/indic_crnn", help="Directory to save model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cpu', 'cuda', or 'auto')")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    train_indic_crnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        resume_path=args.resume,
        device_name=args.device,
    )
