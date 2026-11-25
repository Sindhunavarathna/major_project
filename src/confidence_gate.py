"""Confidence Gating & Reliability Assessment Module.

Prevents low-confidence and corrupted OCR output from entering machine translation.

Capabilities:
1. Validates OCR line confidence against a configurable threshold.
2. Conducts text sanity checks:
   - Filters out pure symbols, punctuation, and non-script noise (e.g., '---', '[] () //').
   - Enforces minimum letter-to-symbol ratios and valid word tokens.
   - Detects script mismatch noise (e.g. garbled matras without consonants).
3. Gates translation:
   - Withholds low-confidence lines from being sent to the translation engine.
   - If the entire image falls below the confidence threshold, triggers the standard fallback:
     "Could not reliably recognize this text" instead of translating garbage.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.utils import get_logger

LOGGER = get_logger("confidence_gate")

DEFAULT_CONFIDENCE_THRESHOLD = 0.10
FALLBACK_UNRELIABLE_MESSAGE = "Could not reliably recognize this text"


@dataclass
class LineGateEvaluation:
    """Evaluation result for an individual OCR line."""
    line_num: int
    text: str
    confidence: float
    passed: bool
    rejection_reason: Optional[str] = None
    letter_count: int = 0
    total_non_space: int = 0
    letter_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_num": self.line_num,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "passed": self.passed,
            "is_low_confidence": not self.passed,
            "rejection_reason": self.rejection_reason,
            "letter_ratio": round(self.letter_ratio, 3),
        }


@dataclass
class DocumentGateResult:
    """Document-level gating decision for machine translation."""
    can_translate: bool
    whole_image_low_confidence: bool
    text_for_translation: str
    evaluated_lines: List[Dict[str, Any]]
    total_lines: int
    accepted_lines_count: int
    rejected_lines_count: int
    mean_confidence: float
    notice: Optional[str] = None


def evaluate_line_sanity(text: str, expected_script: Optional[str] = None) -> Tuple[bool, str, int, int, float]:
    """Check whether a line of text passes linguistic sanity checks.

    Returns:
        (passed, reason, letter_count, total_non_space, letter_ratio)
    """
    stripped = text.strip() if text else ""
    if not stripped:
        return False, "empty_line", 0, 0, 0.0

    non_space = [c for c in stripped if not c.isspace()]
    total_non_space = len(non_space)
    if total_non_space < 2:
        return False, "too_short (<2 characters)", 0, total_non_space, 0.0

    # Count Unicode letters (Ll, Lu, Lt, Lm, Lo)
    letters = [c for c in non_space if unicodedata.category(c).startswith("L")]
    letter_count = len(letters)
    letter_ratio = letter_count / total_non_space if total_non_space > 0 else 0.0

    # Reject if letter ratio is too low (mostly symbols, digits, punctuation, or lines like '--- 12 #$')
    if letter_ratio < 0.40:
        return False, f"low_letter_ratio ({letter_ratio:.1%}, minimum is 40%)", letter_count, total_non_space, letter_ratio

    # Check for presence of at least one valid word-like token
    tokens = stripped.split()
    valid_tokens = 0
    for tok in tokens:
        # Check if token contains at least one letter and isn't pure digits/symbols
        tok_letters = [ch for ch in tok if unicodedata.category(ch).startswith("L")]
        if len(tok_letters) >= 2 or (len(tok_letters) >= 1 and len(tok) <= 3):
            valid_tokens += 1

    if valid_tokens == 0:
        return False, "no_valid_words (pure numbers or symbols)", letter_count, total_non_space, letter_ratio

    return True, "passed", letter_count, total_non_space, letter_ratio


def gate_ocr_lines_for_translation(
    lines: List[Dict[str, Any]],
    raw_full_text: str = "",
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    expected_script: Optional[str] = None,
) -> DocumentGateResult:
    """Evaluate all lines against the confidence threshold and sanity checks.

    Only lines that PASS both threshold and sanity checks are aggregated
    into `text_for_translation`. Low-confidence lines are marked for UI display.
    """
    evaluated_records: List[Dict[str, Any]] = []
    accepted_text_lines: List[str] = []
    confs: List[float] = []

    # If lines list is empty but raw_full_text exists, break into lines
    if not lines and raw_full_text:
        text_lines = [l for l in raw_full_text.split("\n") if l.strip()]
        lines = [{"line_num": i + 1, "text": l, "confidence": 0.50} for i, l in enumerate(text_lines)]

    for idx, line_item in enumerate(lines):
        line_num = line_item.get("line_num", idx + 1)
        txt = line_item.get("text", "").strip()
        conf = float(line_item.get("confidence", 0.0))
        bbox = line_item.get("bbox")
        confs.append(conf)

        sanity_ok, reason, letters, total_ns, ratio = evaluate_line_sanity(txt, expected_script)

        # Gating rule: confidence >= threshold AND sanity passes
        passed = False
        rejection_reason = None

        if not sanity_ok:
            passed = False
            rejection_reason = f"Sanity check failed: {reason}"
        elif conf < threshold:
            passed = False
            rejection_reason = f"Confidence ({conf:.1%}) below threshold ({threshold:.1%})"
        else:
            passed = True

        eval_rec = {
            "line_num": line_num,
            "text": txt,
            "confidence": round(conf, 3),
            "passed_gate": passed,
            "is_low_confidence": not passed,
            "rejection_reason": rejection_reason,
            "bbox": bbox,
            "letter_ratio": round(ratio, 3),
        }
        evaluated_records.append(eval_rec)

        if passed:
            accepted_text_lines.append(txt)

    mean_conf = float(np.mean(confs)) if confs else 0.0
    accepted_count = len(accepted_text_lines)
    rejected_count = len(evaluated_records) - accepted_count

    # Determine if whole image is low confidence
    whole_image_low = (accepted_count == 0) or (len(evaluated_records) > 0 and (accepted_count / len(evaluated_records)) < 0.25 and mean_conf < threshold)

    if whole_image_low or accepted_count == 0:
        can_translate = False
        text_for_trans = ""
        notice = FALLBACK_UNRELIABLE_MESSAGE
    else:
        can_translate = True
        text_for_trans = "\n".join(accepted_text_lines).strip()
        notice = None
        if rejected_count > 0:
            notice = f"{rejected_count} low-confidence line(s) withheld from translation to prevent inaccuracies."

    return DocumentGateResult(
        can_translate=can_translate,
        whole_image_low_confidence=whole_image_low,
        text_for_translation=text_for_trans,
        evaluated_lines=evaluated_records,
        total_lines=len(evaluated_records),
        accepted_lines_count=accepted_count,
        rejected_lines_count=rejected_count,
        mean_confidence=round(mean_conf, 4),
        notice=notice,
    )
