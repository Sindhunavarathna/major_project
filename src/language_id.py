"""Language Identification for Latin-script handwritten output.

Specialized in distinguishing German (de) vs English (en) for Latin text
using langdetect with robust lexical and character-trigram fallback.
"""

from __future__ import annotations

import re
from typing import Tuple
from src.utils import get_logger

LOGGER = get_logger("language_id")

# High-frequency German function words and orthographic markers
GERMAN_MARKERS = {
    "der", "die", "das", "und", "ist", "in", "den", "von", "zu", "mit",
    "sich", "des", "auf", "für", "im", "dem", "nicht", "ein", "eine", "als",
    "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach",
    "wird", "bei", "einer", "um", "am", "sind", "noch", "wie", "einem", "über",
    "einen", "so", "war", "haben", "nur", "oder", "aber", "vor", "zur", "bis"
}

ENGLISH_MARKERS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"
}


def distinguish_german_vs_english(text: str) -> Tuple[str, float]:
    """Classify Latin-script text as German ('de') or English ('en').

    Args:
        text: Input Latin string.

    Returns:
        Tuple of (lang_code, confidence) where lang_code is 'de' or 'en'.
    """
    if not text or not text.strip():
        return "en", 0.50

    cleaned = text.strip()

    # Rule 1: Distinctive German Umlauts / Eszett (ä, ö, ü, ß)
    if any(c in cleaned for c in "äöüßÄÖÜẞ"):
        LOGGER.info("German umlaut detected in Latin text -> German ('de')")
        return "de", 0.95

    # Rule 2: Token-level functional vocabulary match
    tokens = [t.lower() for t in re.findall(r"\b[a-zA-ZäöüßÄÖÜẞ]+\b", cleaned)]
    if tokens:
        de_hits = sum(1 for t in tokens if t in GERMAN_MARKERS)
        en_hits = sum(1 for t in tokens if t in ENGLISH_MARKERS)
        if de_hits > en_hits:
            conf = min(0.95, 0.60 + 0.10 * (de_hits - en_hits))
            return "de", conf
        elif en_hits > de_hits:
            conf = min(0.95, 0.60 + 0.10 * (en_hits - de_hits))
            return "en", conf

    # Rule 3: langdetect statistical detection
    try:
        from langdetect import detect_langs
        predictions = detect_langs(cleaned)
        for pred in predictions:
            if pred.lang == "de":
                return "de", float(pred.prob)
            elif pred.lang == "en":
                return "en", float(pred.prob)
        # Default to highest if Latin
        if predictions:
            top_lang = predictions[0].lang
            if top_lang == "de":
                return "de", float(predictions[0].prob)
    except Exception as e:
        LOGGER.debug("langdetect fallback failed: %s", e)

    # Default fallback is English
    return "en", 0.60
