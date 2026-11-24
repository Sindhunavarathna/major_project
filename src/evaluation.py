"""Comprehensive Evaluation Module: Baseline vs Our Custom Deep Learning Models.

Computes CER (Character Error Rate) and WER (Word Error Rate) per language
on held-out test sets (strictly writer-separated), outputs results table
(Markdown + CSV), and generates comparative visual bar charts.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.utils import get_logger, load_image

LOGGER = get_logger("evaluation")


def evaluate_baseline_vs_our_model(
    output_dir: str = "docs",
    checkpoint_path: str = "models/indic_crnn/best_indic_crnn.pth",
    seed: int = 42,
) -> Dict[str, dict]:
    """Execute end-to-end evaluation comparing Baseline (EasyOCR) vs Our Shared Indic CRNN."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset samples
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

    # Split strictly by writer with zero test leakage
    _, _, test_samples, split_counts = split_samples_by_writer(
        samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )

    LOGGER.info("Evaluation running on %d test samples (writer-isolated)", len(test_samples))

    # 2. Initialize engines
    our_engine = IndicCRNNEngine(checkpoint_path=checkpoint_path)
    easy_engine = EasyOCREngine()

    lang_codes = {"Kannada": "kn", "Tamil": "ta", "Telugu": "te"}
    results: Dict[str, dict] = {}
    table_rows = []

    languages = ["Kannada", "Tamil", "Telugu"]
    for lang in languages:
        lang_samples = [s for s in test_samples if s.language == lang]
        if not lang_samples:
            continue

        lcode = lang_codes.get(lang, "en")
        refs = [s.text for s in lang_samples]

        # Evaluate Baseline
        easy_preds = []
        for s in lang_samples:
            img = s.image_array if s.image_array is not None else s.image_path
            res = easy_engine.recognize(img, lang=lcode)
            easy_preds.append(res.text)

        easy_cer, easy_wer = compute_corpus_metrics(refs, easy_preds)

        # Evaluate Our Model
        our_preds = []
        for s in lang_samples:
            img = s.image_array if s.image_array is not None else s.image_path
            res = our_engine.recognize(img, lang=lcode)
            our_preds.append(res.text)

        our_cer, our_wer = compute_corpus_metrics(refs, our_preds)

        results[lang] = {
            "samples": len(lang_samples),
            "baseline": {"cer": easy_cer, "wer": easy_wer},
            "our_model": {"cer": our_cer, "wer": our_wer},
        }

        table_rows.append({
            "Language": lang,
            "Test_Samples": len(lang_samples),
            "Baseline_CER": f"{easy_cer * 100:.2f}%",
            "Baseline_WER": f"{easy_wer * 100:.2f}%",
            "Our_Model_CER": f"{our_cer * 100:.2f}%",
            "Our_Model_WER": f"{our_wer * 100:.2f}%",
        })

    # Macro averages
    all_refs = [s.text for s in test_samples]
    all_easy_preds = []
    all_our_preds = []
    for s in test_samples:
        lcode = lang_codes.get(s.language, "en")
        img = s.image_array if s.image_array is not None else s.image_path
        all_easy_preds.append(easy_engine.recognize(img, lang=lcode).text)
        all_our_preds.append(our_engine.recognize(img, lang=lcode).text)

    macro_easy_cer, macro_easy_wer = compute_corpus_metrics(all_refs, all_easy_preds)
    macro_our_cer, macro_our_wer = compute_corpus_metrics(all_refs, all_our_preds)

    table_rows.append({
        "Language": "OVERALL (Macro-Avg)",
        "Test_Samples": len(test_samples),
        "Baseline_CER": f"{macro_easy_cer * 100:.2f}%",
        "Baseline_WER": f"{macro_easy_wer * 100:.2f}%",
        "Our_Model_CER": f"{macro_our_cer * 100:.2f}%",
        "Our_Model_WER": f"{macro_our_wer * 100:.2f}%",
    })

    # 3. Save CSV
    csv_file = os.path.join(output_dir, "evaluation_results.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Language", "Test_Samples", "Baseline_CER", "Baseline_WER", "Our_Model_CER", "Our_Model_WER"]
        )
        writer.writeheader()
        writer.writerows(table_rows)

    # 4. Save Markdown Table
    md_file = os.path.join(output_dir, "evaluation_results.md")
    md_lines = [
        "# Systematic Performance Benchmark: Baseline vs Our Custom Deep Learning Models\n",
        f"**Test Set:** {len(test_samples)} held-out manuscript word samples (writer-isolated, zero data leakage)\n",
        "| Language | Test Samples | Baseline CER (EasyOCR) | Baseline WER (EasyOCR) | Our Model CER (Indic-CRNN) | Our Model WER (Indic-CRNN) |",
        "|---|---|---|---|---|---|",
    ]
    for row in table_rows:
        md_lines.append(
            f"| **{row['Language']}** | {row['Test_Samples']} | {row['Baseline_CER']} | {row['Baseline_WER']} | {row['Our_Model_CER']} | {row['Our_Model_WER']} |"
        )
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # 5. Generate Comparative Bar Chart Figure
    chart_file = os.path.join(output_dir, "evaluation_benchmark_chart.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    plot_langs = [l for l in languages if l in results]
    x = np.arange(len(plot_langs))
    bar_width = 0.35

    # Plot CER
    base_cers = [results[l]["baseline"]["cer"] * 100 for l in plot_langs]
    our_cers = [results[l]["our_model"]["cer"] * 100 for l in plot_langs]

    ax1.bar(x - bar_width / 2, base_cers, bar_width, label="Baseline (EasyOCR)", color="#94A3B8")
    ax1.bar(x + bar_width / 2, our_cers, bar_width, label="Our Shared Indic-CRNN", color="#38BDF8")
    ax1.set_title("Character Error Rate (CER) per Language\n(Lower is Better)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("CER (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_langs, fontsize=10)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Plot WER
    base_wers = [results[l]["baseline"]["wer"] * 100 for l in plot_langs]
    our_wers = [results[l]["our_model"]["wer"] * 100 for l in plot_langs]

    ax2.bar(x - bar_width / 2, base_wers, bar_width, label="Baseline (EasyOCR)", color="#94A3B8")
    ax2.bar(x + bar_width / 2, our_wers, bar_width, label="Our Shared Indic-CRNN", color="#10B981")
    ax2.set_title("Word Error Rate (WER) per Language\n(Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("WER (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_langs, fontsize=10)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(chart_file, bbox_inches="tight")
    plt.close(fig)

    LOGGER.info("Saved evaluation CSV to %s, Markdown to %s, and Chart to %s", csv_file, md_file, chart_file)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs Our Custom Models")
    parser.add_argument("--output-dir", type=str, default="docs", help="Output directory")
    args = parser.parse_args()
    evaluate_baseline_vs_our_model(output_dir=args.output_dir)
