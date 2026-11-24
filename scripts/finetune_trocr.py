"""Fine-tuning TrOCR (VisionEncoderDecoderModel) on Manuscript Image-Text Dataset.

NOTE: Designed to run on Google Colab / GPU accelerator.
Saves fine-tuned weights to models/sanskrit_trocr/ for direct ingestion by the system.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)


class ManuscriptDataset(Dataset):
    """Dataset loading line image crops and corresponding transcription text."""

    def __init__(self, df: pd.DataFrame, image_dir: str, processor: TrOCRProcessor, max_target_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_name = row["image_file"]
        text = str(row["text"])

        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id's of the label by -100 so it doesn't compute loss
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
        }


def compute_metrics(pred, processor):
    """Compute Character Error Rate (CER) and Word Error Rate (WER)."""
    import jiwer

    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = jiwer.cer(label_str, pred_str)
    wer = jiwer.wer(label_str, pred_str)

    return {"cer": cer, "wer": wer}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on manuscript lines")
    parser.add_argument("--data_csv", type=str, default="data/train_lines.csv", help="CSV with columns: image_file, text")
    parser.add_argument("--image_dir", type=str, default="data/line_images", help="Directory holding line crops")
    parser.add_argument("--model_name", type=str, default="microsoft/trocr-base-handwritten")
    parser.add_argument("--output_dir", type=str, default="models/sanskrit_trocr")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing TrOCR fine-tuning on device: {device}")

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    # Set special token IDs for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Freeze vision encoder backbone initially if dataset is small
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    if os.path.exists(args.data_csv):
        df = pd.read_csv(args.data_csv)
        train_df = df.sample(frac=0.85, random_state=42)
        val_df = df.drop(train_df.index)

        train_dataset = ManuscriptDataset(train_df, args.image_dir, processor)
        val_dataset = ManuscriptDataset(val_df, args.image_dir, processor)
    else:
        print(f"Warning: Dataset CSV '{args.data_csv}' not found. Please provide a valid dataset.")
        return

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=torch.cuda.is_available(),
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=lambda p: compute_metrics(p, processor),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuned model successfully saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
