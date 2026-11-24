"""Evaluation and benchmarking script.

Computes:
- CER (Character Error Rate) and WER (Word Error Rate) via jiwer for OCR engines.
- BLEU and chrF scores via sacrebleu for translated texts.
Evaluates on sample_data/ ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path so scripts can run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List
import jiwer
import sacrebleu

from src.pipeline import ManuscriptPipeline


def run_evaluation():
    sample_dir = Path(__file__).resolve().parent.parent / "sample_data"
    samples_file = sample_dir / "samples.json"

    if not samples_file.exists():
        print(f"Error: {samples_file} not found.")
        return

    with open(samples_file, "r", encoding="utf-8") as f:
        ground_truth_samples = json.load(f)

    print("=" * 70)
    print("MANUSCRIPT TRANSLATOR BENCHMARK & EVALUATION REPORT")
    print("=" * 70)

    pipeline = ManuscriptPipeline()

    ocr_references: List[str] = []
    ocr_hypotheses: List[str] = []

    trans_references: List[str] = []
    trans_hypotheses: List[str] = []

    print(f"\nEvaluating {len(ground_truth_samples)} sample documents...\n")

    for item in ground_truth_samples:
        filename = item["filename"]
        img_path = sample_dir / filename
        gt_text = item["text"]
        gt_trans = item["translation_en"]
        lang = item["lang"]

        if not img_path.exists():
            print(f"Skipping missing sample: {filename}")
            continue

        # Run pipeline
        res = pipeline.process_image(
            image_input=str(img_path),
            mode="Auto",
            target_lang="en",
        )

        pred_ocr = res.ocr_result.text.strip()
        pred_trans = res.translation_result.translated_text.strip() if res.translation_result else ""

        # Compute individual metrics
        cer_item = jiwer.cer(gt_text, pred_ocr) if pred_ocr else 1.0
        wer_item = jiwer.wer(gt_text, pred_ocr) if pred_ocr else 1.0

        ocr_references.append(gt_text)
        ocr_hypotheses.append(pred_ocr if pred_ocr else " ")

        if gt_trans and pred_trans:
            trans_references.append(gt_trans)
            trans_hypotheses.append(pred_trans)

        print(f"File: {filename} [{lang.upper()}]")
        print(f"  Ground Truth Text : {gt_text}")
        print(f"  Recognized Text   : {pred_ocr}")
        print(f"  OCR CER: {cer_item * 100:.2f}% | WER: {wer_item * 100:.2f}%")
        print(f"  Language Detected : {res.langid_result.language_name} ({res.langid_result.confidence:.1%})")
        print(f"  Translation (EN)  : {pred_trans}")
        print("-" * 70)

    # Aggregate OCR metrics
    overall_cer = jiwer.cer(ocr_references, ocr_hypotheses)
    overall_wer = jiwer.wer(ocr_references, ocr_hypotheses)

    print("\n" + "=" * 70)
    print("AGGREGATE OCR PERFORMANCE (jiwer):")
    print(f"  Mean Character Error Rate (CER): {overall_cer * 100:.2f}%")
    print(f"  Mean Word Error Rate (WER)     : {overall_wer * 100:.2f}%")
    print("=" * 70)

    # Aggregate Translation metrics
    if trans_hypotheses and trans_references:
        formatted_refs = [[ref] for ref in trans_references]
        bleu = sacrebleu.corpus_bleu(trans_hypotheses, formatted_refs)
        chrf = sacrebleu.corpus_chrf(trans_hypotheses, formatted_refs)

        print("\nAGGREGATE TRANSLATION PERFORMANCE (sacrebleu):")
        print(f"  BLEU Score : {bleu.score:.2f}")
        print(f"  chrF2 Score: {chrf.score:.2f}")
        print("=" * 70)
    else:
        print("\nTranslation evaluation skipped: No translation outputs available.")


if __name__ == "__main__":
    run_evaluation()
