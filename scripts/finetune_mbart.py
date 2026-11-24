"""Fine-tune mBART-50 on parallel translation corpora (e.g. Sanskrit/Hindi/Kannada/English).

NOTE: Designed to run on Google Colab / GPU accelerator.
Loads a parallel CSV with columns: source_text, target_text, src_lang, tgt_lang
Saves fine-tuned model checkpoint for enhanced historical translation.
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
import torch
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


class ParallelTranslationDataset(Dataset):
    """Dataset for parallel sentence pairs with source and target language markers."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: MBart50TokenizerFast,
        max_source_length: int = 128,
        max_target_length: int = 128,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        src_text = str(row["source_text"])
        tgt_text = str(row["target_text"])
        src_lang = str(row.get("src_lang", "hi_IN"))
        # Set tokenizer language codes
        self.tokenizer.src_lang = src_lang
        self.tokenizer.src_lang = src_lang
        model_inputs = self.tokenizer(
            src_text,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_text,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"]

        labels[labels == self.tokenizer.pad_token_id] = -100

        item = {key: val.squeeze(0) for key, val in model_inputs.items()}
        item["labels"] = labels.squeeze(0)
        return item


def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU and chrF scores using sacrebleu."""
    import sacrebleu

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = [[tok for tok in label if tok != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels)

    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune mBART-50 for manuscript translation")
    parser.add_argument("--data_csv", type=str, default="data/parallel_corpus.csv")
    parser.add_argument("--model_name", type=str, default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_mbart")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing mBART-50 fine-tuning on device: {device}")

    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_name)
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)

    if not os.path.exists(args.data_csv):
        print(f"Warning: Training CSV '{args.data_csv}' not found. Please provide a parallel CSV.")
        return

    df = pd.read_csv(args.data_csv)
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = ParallelTranslationDataset(train_df, tokenizer)
    val_dataset = ParallelTranslationDataset(val_df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Fine-tuned translation model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
