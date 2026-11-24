"""Two-tier Language Identification module combining Unicode script detection and Naive Bayes n-grams.

Features:
- Unicode script range detection (Latin, Devanagari, Kannada, Tamil, Telugu, Arabic, etc.).
- Script-constrained Multinomial Naive Bayes classifier trained on character n-grams (1-3).
- Classical orthographic and morphological disambiguation (e.g. Sanskrit vs Hindi).
- Secondary validation method using langdetect for side-by-side comparison.
- Uncertainty gating: marks short or low-confidence (<50%) or symbol inputs as 'Uncertain'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib

from config import CONFIG
from src.utils import get_logger

LOGGER = get_logger("langid")

# Unicode block definitions for script classification
UNICODE_SCRIPTS = {
    "Devanagari": (0x0900, 0x097F),
    "Kannada": (0x0C80, 0x0CFF),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Arabic": (0x0600, 0x06FF),
    "Bengali": (0x0980, 0x09FF),
    "Gujarati": (0x0A80, 0x0AFF),
    "Latin": [
        (0x0041, 0x005A),  # Basic Latin uppercase
        (0x0061, 0x007A),  # Basic Latin lowercase
        (0x00C0, 0x00FF),  # Latin-1 Supplement
        (0x0100, 0x017F),  # Latin Extended-A
    ],
}

# Mapping of scripts to valid languages
SCRIPT_TO_LANGUAGES = {
    "Devanagari": ["hi", "sa", "mr", "ne"],
    "Kannada": ["kn"],
    "Tamil": ["ta"],
    "Telugu": ["te"],
    "Arabic": ["ar", "ur", "fa"],
    "Bengali": ["bn"],
    "Gujarati": ["gu"],
    "Latin": ["en", "de", "la", "fr", "es"],
}

# Language display names
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "sa": "Sanskrit",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ar": "Arabic",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ur": "Urdu",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "la": "Latin",
    "ne": "Nepali",
}


@dataclass
class CandidatePrediction:
    language: str
    language_name: str
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "language_name": self.language_name,
            "probability": round(self.probability, 4),
        }


@dataclass
class LanguageIDResult:
    """Standardized result of language and script identification."""
    script: str
    script_confidence: float
    language: str
    language_name: str
    confidence: float
    is_uncertain: bool
    top_candidates: List[CandidatePrediction] = field(default_factory=list)
    secondary_method: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "script": self.script,
            "script_confidence": round(self.script_confidence, 4),
            "language": self.language,
            "language_name": self.language_name,
            "confidence": round(self.confidence, 4),
            "is_uncertain": self.is_uncertain,
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "secondary_method": self.secondary_method,
            "notes": self.notes,
        }


def detect_unicode_script(text: str) -> Tuple[str, float, Dict[str, int]]:
    """Determine dominant Unicode script in text and script confidence ratio."""
    counts: Dict[str, int] = {k: 0 for k in UNICODE_SCRIPTS}
    total_letters = 0

    for ch in text:
        cp = ord(ch)
        for script_name, ranges in UNICODE_SCRIPTS.items():
            if isinstance(ranges, list):
                if any(start <= cp <= end for start, end in ranges):
                    counts[script_name] += 1
                    total_letters += 1
                    break
            else:
                start, end = ranges
                if start <= cp <= end:
                    counts[script_name] += 1
                    total_letters += 1
                    break

    if total_letters == 0:
        return "Unknown", 0.0, counts

    dominant_script = max(counts, key=counts.get)
    script_conf = counts[dominant_script] / total_letters

    if counts[dominant_script] == 0:
        return "Unknown", 0.0, counts

    return dominant_script, script_conf, counts


class LanguageIdentifier:
    """Hybrid two-tier language identifier."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(
            model_path or CONFIG.language_classifier.model_path
        )
        self._pipeline: Any = None
        self._classes: List[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Load trained Naive Bayes classifier if available."""
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self._pipeline = data.get("pipeline")
                self._classes = list(data.get("classes", []))
                LOGGER.info("Language classifier loaded from %s.", self.model_path)
            except Exception as e:
                LOGGER.warning("Could not load language classifier from %s: %s", self.model_path, e)
                self._pipeline = None
        else:
            LOGGER.warning("Language classifier not found at %s.", self.model_path)

    def _run_secondary_method(self, text: str) -> Optional[Dict[str, Any]]:
        """Run langdetect as secondary comparison method."""
        try:
            from langdetect import detect_langs
            langs = detect_langs(text)
            if langs:
                top = langs[0]
                lang_code = top.lang
                prob = float(top.prob)
                return {
                    "method": "langdetect",
                    "language": lang_code,
                    "language_name": LANGUAGE_NAMES.get(lang_code, lang_code.upper()),
                    "confidence": round(prob, 4),
                }
        except Exception:
            pass
        return None

    def _refine_devanagari(self, text: str, initial_probs: Dict[str, float]) -> Dict[str, float]:
        """Apply linguistic heuristics to disambiguate Sanskrit vs Hindi in Devanagari."""
        scores = dict(initial_probs)
        clean = text.strip()

        # Sanskrit markers: Visarga (ः), Avagraha (ऽ), danda (।), typical endings
        sanskrit_cues = 0
        if "ः" in clean:
            sanskrit_cues += 3
        if "ऽ" in clean:
            sanskrit_cues += 3
        if "।" in clean or "॥" in clean:
            sanskrit_cues += 1

        # Common Sanskrit case/verbal suffixes
        for marker in ("सु ", "स्य ", "म् ", "आत् ", "एण ", "भ्याम् ", "असि ", "अस्ति ", "भवन्ति "):
            if marker in clean:
                sanskrit_cues += 2

        # Hindi markers: auxiliary verbs and postpositions
        hindi_cues = 0
        for marker in (" है", " हैं", " था", " थी", " थे", " का ", " की ", " के ", " में ", " पर ", " से ", " को "):
            if marker in clean:
                hindi_cues += 2

        if sanskrit_cues > hindi_cues:
            scores["sa"] = scores.get("sa", 0.0) + (0.35 * sanskrit_cues)
        elif hindi_cues > sanskrit_cues:
            scores["hi"] = scores.get("hi", 0.0) + (0.35 * hindi_cues)

        return scores

    def identify(self, text: str) -> LanguageIDResult:
        """Identify language from text combining Unicode script detection and Naive Bayes."""
        stripped = text.strip() if text else ""

        # Step 1: Check length and noise patterns
        # Reject empty, very short, or junk strings like '# 0'
        junk_patterns = {"# 0", "#", "0", "1", "---", "...", "_", "*"}
        if len(stripped) < CONFIG.langid.min_text_len or stripped in junk_patterns:
            return LanguageIDResult(
                script="Unknown",
                script_confidence=0.0,
                language="Uncertain",
                language_name="Uncertain (Short/Junk Input)",
                confidence=0.0,
                is_uncertain=True,
                notes="Input text is too short or contains only symbols/digits.",
            )

        # Step 2: Unicode script detection
        script, script_conf, counts = detect_unicode_script(stripped)

        if script == "Unknown" or script_conf < 0.3:
            return LanguageIDResult(
                script="Unknown",
                script_confidence=script_conf,
                language="Uncertain",
                language_name="Uncertain (Undetected Script)",
                confidence=0.0,
                is_uncertain=True,
                notes="No recognizable Unicode script detected.",
            )

        # Step 3: Run Naive Bayes classifier
        raw_probs: Dict[str, float] = {}

        if self._pipeline is not None:
            try:
                proba = self._pipeline.predict_proba([stripped])[0]
                for cls_name, p in zip(self._classes, proba):
                    raw_probs[cls_name] = float(p)
            except Exception as e:
                LOGGER.error("Naive Bayes inference error: %s", e)

        # Script-constraint filtering
        valid_langs = SCRIPT_TO_LANGUAGES.get(script, [])
        if valid_langs and raw_probs:
            # Filter and renormalize probabilities
            filtered = {k: raw_probs.get(k, 0.0001) for k in valid_langs}
            if script == "Devanagari":
                filtered = self._refine_devanagari(stripped, filtered)

            total_p = sum(filtered.values())
            if total_p > 0:
                normalized = {k: v / total_p for k, v in filtered.items()}
            else:
                normalized = {k: 1.0 / len(valid_langs) for k in valid_langs}
        elif raw_probs:
            normalized = raw_probs
        elif valid_langs:
            # Default to primary language of the script if model unavailable
            normalized = {valid_langs[0]: 1.0}
        else:
            normalized = {"en": 1.0}

        sorted_candidates = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        top_candidates = [
            CandidatePrediction(
                language=lang_code,
                language_name=LANGUAGE_NAMES.get(lang_code, lang_code.upper()),
                probability=round(p, 4),
            )
            for lang_code, p in sorted_candidates[:3]
        ]

        top_lang = top_candidates[0].language if top_candidates else "en"
        top_prob = top_candidates[0].probability if top_candidates else 0.0

        # Secondary method
        secondary = self._run_secondary_method(stripped)

        # Step 4: Quality & Uncertainty checks
        threshold = CONFIG.langid.confidence_threshold
        is_uncertain = (top_prob < threshold) or (len(stripped) < 4)

        chosen_lang = "Uncertain" if is_uncertain else top_lang
        chosen_name = "Uncertain" if is_uncertain else LANGUAGE_NAMES.get(top_lang, top_lang.upper())

        notes = None
        if is_uncertain:
            notes = (
                f"Confidence ({top_prob:.1%}) is below threshold ({threshold:.1%}). "
                "Manual language selection is required."
            )

        return LanguageIDResult(
            script=script,
            script_confidence=script_conf,
            language=chosen_lang,
            language_name=chosen_name,
            confidence=top_prob,
            is_uncertain=is_uncertain,
            top_candidates=top_candidates,
            secondary_method=secondary,
            notes=notes,
        )


IDENTIFIER = LanguageIdentifier()


def identify_language(text: str) -> LanguageIDResult:
    """Convenience functional interface for language identification."""
    return IDENTIFIER.identify(text)

