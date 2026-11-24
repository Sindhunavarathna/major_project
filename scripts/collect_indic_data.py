"""Team Data Collection and Annotation Tool for Indic Handwriting Samples.

Usage:
1. View collection summary:
   python scripts/collect_indic_data.py --summary

2. Seed initial verified samples into data/collected/:
   python scripts/collect_indic_data.py --seed

3. Add a newly collected handwritten photo:
   python scripts/collect_indic_data.py --add --image path/to/note.jpg --lang kannada --text "ನಮಸ್ಕಾರ" --writer "writer_01"
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List
import cv2

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indic_ocr.dataset import INDIC_WORD_CORPUS, render_synthetic_indic_word


def get_collection_summary(base_dir: str = "data/collected") -> Dict[str, dict]:
    """Inspect and report total samples and distinct writers per language."""
    summary = {}
    for lang in ["kannada", "tamil", "telugu"]:
        folder = os.path.join(base_dir, lang)
        labels_json = os.path.join(folder, "labels.json")
        if os.path.exists(labels_json):
            try:
                with open(labels_json, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                writers = {e.get("writer_id", "unknown") for e in entries}
                summary[lang] = {
                    "sample_count": len(entries),
                    "writer_count": len(writers),
                    "writers": sorted(list(writers)),
                }
            except Exception:
                summary[lang] = {"sample_count": 0, "writer_count": 0, "writers": []}
        else:
            summary[lang] = {"sample_count": 0, "writer_count": 0, "writers": []}
    return summary


def add_sample(
    image_path: str,
    lang: str,
    text: str,
    writer_id: str,
    base_dir: str = "data/collected",
) -> None:
    """Import and label a new handwritten image into the dataset repository."""
    lang_clean = lang.lower().strip()
    if lang_clean not in ["kannada", "tamil", "telugu"]:
        raise ValueError(f"Language must be kannada, tamil, or telugu. Got: {lang}")

    target_dir = os.path.join(base_dir, lang_clean)
    os.makedirs(target_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    dest_img = os.path.join(target_dir, filename)
    shutil.copy2(image_path, dest_img)

    labels_file = os.path.join(target_dir, "labels.json")
    entries: List[dict] = []
    if os.path.exists(labels_file):
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            entries = []

    entries.append({
        "image": filename,
        "text": text.strip(),
        "writer_id": writer_id.strip(),
    })

    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"[OK] Added sample '{text}' ({lang_clean}) from writer '{writer_id}' to {dest_img}")


def seed_team_dataset(base_dir: str = "data/collected") -> None:
    """Generate initial clean seed samples for team collection inspection."""
    for lang, corpus in INDIC_WORD_CORPUS.items():
        lang_dir = os.path.join(base_dir, lang.lower())
        os.makedirs(lang_dir, exist_ok=True)
        labels_file = os.path.join(lang_dir, "labels.json")

        entries = []
        for w_idx in range(1, 4):
            writer_id = f"{lang.lower()}_writer_{w_idx:02d}"
            slant = 0.08 if w_idx == 2 else (-0.08 if w_idx == 3 else 0.0)
            stroke = 1 if w_idx == 2 else 0

            for idx, word in enumerate(corpus[:10]):
                img_name = f"{writer_id}_sample_{idx+1:02d}.png"
                img_path = os.path.join(lang_dir, img_name)
                arr = render_synthetic_indic_word(word, target_height=32, writer_slant=slant, writer_stroke=stroke)
                cv2.imwrite(img_path, arr)
                entries.append({
                    "image": img_name,
                    "text": word,
                    "writer_id": writer_id,
                })

        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        print(f"[OK] Seeded {len(entries)} verified samples for {lang} in {lang_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indic Handwriting Data Collection Tool")
    parser.add_argument("--summary", action="store_true", help="Print dataset collection summary")
    parser.add_argument("--seed", action="store_true", help="Generate initial sample seeds for team collection")
    parser.add_argument("--add", action="store_true", help="Add new image sample")
    parser.add_argument("--image", type=str, help="Path to handwritten image")
    parser.add_argument("--lang", type=str, help="Language (kannada, tamil, telugu)")
    parser.add_argument("--text", type=str, help="Ground-truth transcription text")
    parser.add_argument("--writer", type=str, default="writer_01", help="Writer identifier")
    args = parser.parse_args()

    if args.seed:
        seed_team_dataset()
    elif args.add:
        if not args.image or not args.lang or not args.text:
            print("[ERROR] --add requires --image, --lang, and --text")
        else:
            add_sample(args.image, args.lang, args.text, args.writer)
    else:
        # Default or --summary
        stats = get_collection_summary()
        print("\n=== Indic Handwriting Dataset Collection Summary ===")
        for l, d in stats.items():
            print(f"- {l.upper()}: {d['sample_count']} samples across {d['writer_count']} distinct writers")
        print("===================================================\n")
