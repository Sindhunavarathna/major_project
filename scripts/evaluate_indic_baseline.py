"""Baseline Comparison Script: EasyOCR Baseline vs. Shared Indic CRNN.

Evaluates on the held-out test split (zero writer leakage) across
Kannada, Tamil, and Telugu scripts, computing CER and WER.
Outputs results table in Markdown and CSV formats.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indic_ocr.charset import UnifiedIndicCharset
from src.indic_ocr.dataset import (
    IndicSample,
    generate_seed_indic_samples,
    load_iiit_indic_samples,
    load_team_collected_samples,
    split_samples_by_writer,
)
from src.indic_ocr.metrics import compute_cer, compute_corpus_metrics, compute_wer
from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.indic_crnn_engine import IndicCRNNEngine
from src.utils import get_logger
import cv2
import numpy as np

LOGGER = get_logger("evaluate_indic_baseline")


def _get_image_array(sample: IndicSample) -> np.ndarray:
    if sample.image_array is not None:
        return sample.image_array
    if sample.image_path and os.path.exists(sample.image_path):
        arr = cv2.imread(sample.image_path)
        if arr is not None:
            return arr
    return np.full((32, 64, 3), 255, dtype=np.uint8)


def run_indic_benchmark(
    checkpoint_path: str = "models/indic_crnn/best_indic_crnn.pth",
    output_dir: str = "docs",
    seed: int = 42,
) -> Dict[str, dict]:
    """Run comparative benchmark: Baseline (EasyOCR) vs. Our Shared Indic-CRNN."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load test samples
    samples: List[IndicSample] = []
    iiit_samples = load_iiit_indic_samples("data")
    if iiit_samples:
        samples.extend(iiit_samples)
    team_samples = load_team_collected_samples("data/collected")
    if team_samples:
        samples.extend(team_samples)
    if len(samples) < 90:
        seed_samples = generate_seed_indic_samples(writers_per_lang=6, words_per_writer=15, seed=seed)
        samples.extend(seed_samples)

    _, _, test_samples, split_counts = split_samples_by_writer(
        samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )

    LOGGER.info("Held-out test set size: %d samples across %d languages", len(test_samples), len(split_counts))

    # 2. Initialize engines
    our_crnn = IndicCRNNEngine(checkpoint_path=checkpoint_path)
    easyocr_engine = EasyOCREngine()

    lang_code_map = {"Kannada": "kn", "Tamil": "ta", "Telugu": "te"}
    results_by_lang: Dict[str, dict] = {}

    table_rows = []

    for lang in ["Kannada", "Tamil", "Telugu"]:
        lang_samples = [s for s in test_samples if s.language == lang]
        if not lang_samples:
            continue

        lcode = lang_code_map.get(lang, "en")
        ref_texts = [s.text for s in lang_samples]

        # Evaluate Baseline (EasyOCR)
        easy_preds = []
        for s in lang_samples:
            img = _get_image_array(s)
            res = easyocr_engine.recognize(img, lang=lcode)
            easy_preds.append(res.text)

        easy_cer, easy_wer = compute_corpus_metrics(ref_texts, easy_preds)

        # Evaluate Our Model (Indic CRNN)
        our_preds = []
        for s in lang_samples:
            img = _get_image_array(s)
            res = our_crnn.recognize(img, lang=lcode)
            our_preds.append(res.text)

        our_cer, our_wer = compute_corpus_metrics(ref_texts, our_preds)

        results_by_lang[lang] = {
            "test_samples": len(lang_samples),
            "baseline_easyocr": {"cer": easy_cer, "wer": easy_wer},
            "our_indic_crnn": {"cer": our_cer, "wer": our_wer},
        }

        table_rows.append({
            "Language": lang,
            "Samples": len(lang_samples),
            "Baseline_CER": f"{easy_cer * 100:.2f}%",
            "Baseline_WER": f"{easy_wer * 100:.2f}%",
            "OurModel_CER": f"{our_cer * 100:.2f}%",
            "OurModel_WER": f"{our_wer * 100:.2f}%",
        })

    # Overall Macro-Average
    all_refs = [s.text for s in test_samples]
    total_samples = len(test_samples)
    all_easy_preds = []
    all_our_preds = []
    for s in test_samples:
        lcode = lang_code_map.get(s.language, "en")
        img = _get_image_array(s)
        all_easy_preds.append(easyocr_engine.recognize(img, lang=lcode).text)
        all_our_preds.append(our_crnn.recognize(img, lang=lcode).text)

    overall_easy_cer, overall_easy_wer = compute_corpus_metrics(all_refs, all_easy_preds)
    overall_our_cer, overall_our_wer = compute_corpus_metrics(all_refs, all_our_preds)

    table_rows.append({
        "Language": "OVERALL (Macro-Avg)",
        "Samples": total_samples,
        "Baseline_CER": f"{overall_easy_cer * 100:.2f}%",
        "Baseline_WER": f"{overall_easy_wer * 100:.2f}%",
        "OurModel_CER": f"{overall_our_cer * 100:.2f}%",
        "OurModel_WER": f"{overall_our_wer * 100:.2f}%",
    })

    # 3. Save CSV
    csv_path = os.path.join(output_dir, "indic_recognition_benchmark.csv")
    fieldnames = ["Language", "Samples", "Baseline_CER", "Baseline_WER", "OurModel_CER", "OurModel_WER"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    # 4. Save Markdown
    md_path = os.path.join(output_dir, "indic_recognition_benchmark.md")
    md_content = [
        "# Indic Handwriting Recognition Benchmark: Baseline vs. Shared Indic CRNN\n",
        f"**Test Set:** {total_samples} held-out word samples (writer-separated, zero leakage)\n",
        "| Language | Samples | Baseline CER (EasyOCR) | Baseline WER (EasyOCR) | Our Model CER (Indic-CRNN) | Our Model WER (Indic-CRNN) |",
        "|---|---|---|---|---|---|",
    ]
    for row in table_rows:
        md_content.append(
            f"| **{row['Language']}** | {row['Samples']} | {row['Baseline_CER']} | {row['Baseline_WER']} | {row['OurModel_CER']} | {row['OurModel_WER']} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")

    LOGGER.info("Benchmark complete! Saved CSV to %s and Markdown to %s", csv_path, md_path)
    return results_by_lang


if __name__ == "__main__":
    run_indic_benchmark()
