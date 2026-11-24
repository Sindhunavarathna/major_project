"""OCR quality gating and reliability assessment module."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import CONFIG
from src.utils import get_logger

LOGGER = get_logger("quality_gating")


@dataclass
class QualityCheckResult:
    """Quality gating evaluation result."""
    is_reliable: bool
    can_translate: bool
    char_count: int
    letter_count: int
    letter_ratio: float
    mean_confidence: float
    warning: Optional[str] = None
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_reliable": self.is_reliable,
            "can_translate": self.can_translate,
            "char_count": self.char_count,
            "letter_count": self.letter_count,
            "letter_ratio": round(self.letter_ratio, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "warning": self.warning,
            "details": self.details or {},
        }


def count_letters_and_symbols(text: str, expected_script: Optional[str] = None) -> tuple[int, int]:
    """Count letters versus total non-whitespace characters."""
    total_chars = 0
    letters = 0

    for ch in text:
        if ch.isspace():
            continue
        total_chars += 1
        category = unicodedata.category(ch)
        # Category 'L*' covers all letters in Unicode (Ll, Lu, Lt, Lm, Lo)
        if category.startswith("L"):
            letters += 1

    return letters, total_chars


def evaluate_ocr_quality(
    text: str,
    confidence: float,
    expected_script: Optional[str] = None,
    min_chars: Optional[int] = None,
    min_letter_ratio: Optional[float] = None,
    min_confidence: Optional[float] = None,
    engine_name: Optional[str] = None,
) -> QualityCheckResult:
    """Evaluate whether extracted text meets reliability thresholds for translation.

    Checks:
    1. Text is not empty or whitespace only.
    2. Character count meets minimum length.
    3. Letter-to-character ratio is sufficient (filters out garbage symbols like '# 0', '---').
    4. Model confidence meets the engine-appropriate threshold.

    EasyOCR word-level confidence scores are calibrated differently from TrOCR beam-search
    token probabilities. EasyOCR routinely reports 0.25–0.50 on multi-script content even
    when recognition is good; TrOCR token probabilities on correct output are typically
    0.70–0.99. We therefore apply a lower confidence threshold for EasyOCR results so that
    legitimate EasyOCR fallback output is not blocked by the TrOCR threshold.
    """
    from config import load_config
    cfg = load_config()

    cfg_chars = min_chars if min_chars is not None else cfg.quality_gating.min_characters
    cfg_ratio = min_letter_ratio if min_letter_ratio is not None else cfg.quality_gating.min_letter_ratio

    # Engine-aware confidence threshold
    if min_confidence is not None:
        cfg_conf = min_confidence
    elif engine_name and "easyocr" in engine_name.lower():
        cfg_conf = cfg.quality_gating.easyocr_min_confidence
        LOGGER.debug(
            "Using EasyOCR confidence threshold %.2f (not TrOCR threshold %.2f) for engine '%s'",
            cfg_conf,
            cfg.quality_gating.min_mean_token_confidence,
            engine_name,
        )
    else:
        cfg_conf = cfg.quality_gating.min_mean_token_confidence

    stripped = text.strip() if text else ""
    if not stripped:
        return QualityCheckResult(
            is_reliable=False,
            can_translate=False,
            char_count=0,
            letter_count=0,
            letter_ratio=0.0,
            mean_confidence=0.0,
            warning="OCR output is completely empty.",
            details={"reason": "empty_text"},
        )

    letters, total_non_space = count_letters_and_symbols(stripped, expected_script)
    letter_ratio = (letters / total_non_space) if total_non_space > 0 else 0.0

    # Test for pure symbol or numeric noise (like '# 0')
    is_pure_symbols = letters == 0 or (total_non_space <= 3 and letters <= 1)

    reasons = []
    if total_non_space < cfg_chars:
        reasons.append(f"Text too short ({total_non_space} chars, minimum is {cfg_chars})")

    if letter_ratio < cfg_ratio or is_pure_symbols:
        reasons.append(f"Low letter ratio ({letter_ratio:.1%}, minimum is {cfg_ratio:.1%}) - text contains mostly symbols or numbers")

    if confidence < cfg_conf:
        reasons.append(f"Model confidence ({confidence:.1%}) below threshold ({cfg_conf:.1%})")

    is_reliable = len(reasons) == 0
    can_translate = not is_pure_symbols and total_non_space >= cfg_chars

    warning = None
    if not is_reliable:
        warning = "OCR output looks unreliable. Review and edit the text, or pick a different script/model: " + "; ".join(reasons)

    return QualityCheckResult(
        is_reliable=is_reliable,
        can_translate=can_translate,
        char_count=total_non_space,
        letter_count=letters,
        letter_ratio=letter_ratio,
        mean_confidence=confidence,
        warning=warning,
        details={"reasons": reasons, "is_pure_symbols": is_pure_symbols},
    )

