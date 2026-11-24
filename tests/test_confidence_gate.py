"""Unit tests for OCR Confidence Gating & Reliability Assessment."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app
from src.confidence_gate import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    FALLBACK_UNRELIABLE_MESSAGE,
    evaluate_line_sanity,
    gate_ocr_lines_for_translation,
)
from src.fast_pipeline import FastPipelineManager


def test_evaluate_line_sanity_filters_symbols_and_noise():
    """Verify linguistic sanity filter blocks noise, punctuation, and non-script symbols."""
    # Garbage & Noise lines that MUST fail sanity checks
    garbage_lines = [
        "--- 12 #$",
        ".. ?!",
        "00 00",
        "[] () //",
        "@ # $ %",
        "12345",
        "+++",
        "---",
        "_____",
        ". . .",
        "x",
    ]
    for line in garbage_lines:
        passed, reason, letters, total_ns, ratio = evaluate_line_sanity(line)
        assert not passed, f"Expected '{line}' to fail sanity check, but it passed (reason: {reason})"

    # Legitimate multilingual lines that MUST pass sanity checks
    legitimate_lines = [
        "ಪ್ರಾಚೀನ ಹಸ್ತಪ್ರತಿಗಳು ಭಾರತದ ಇತಿಹಾಸವನ್ನು ತಿಳಿಸುತ್ತವೆ",
        "ಕನ್ನಡ ಸಾಹಿತ್ಯ ಮತ್ತು ಸಂಸ್ಕೃತಿ",
        "பண்டைய கையெழுத்துப் பிரதிகள்",
        "పురాతన రాతప్రతులు సంస్కృతి",
        "Hymel vnd ertreich mere vnd alle",
        "Ancient manuscripts preserve historical knowledge",
        "Die Handschriften des Klosters",
    ]
    for line in legitimate_lines:
        passed, reason, letters, total_ns, ratio = evaluate_line_sanity(line)
        assert passed, f"Expected '{line}' to pass sanity check, but it failed (reason: {reason})"


def test_gate_ocr_lines_filters_low_confidence_lines():
    """Verify that lines below the configurable threshold are marked and excluded from translation."""
    test_lines = [
        {"line_num": 1, "text": "Hymel vnd ertreich", "confidence": 0.85},
        {"line_num": 2, "text": "Fxense 3w0 88*9", "confidence": 0.22},
        {"line_num": 3, "text": "mere vnd alle wasser", "confidence": 0.78},
        {"line_num": 4, "text": "--- 12 #$", "confidence": 0.80},  # Fails sanity check despite high conf
    ]

    result = gate_ocr_lines_for_translation(test_lines, threshold=0.35)

    assert result.can_translate is True
    assert result.accepted_lines_count == 2
    assert result.rejected_lines_count == 2

    # Line 1 and 3 should pass
    assert result.evaluated_lines[0]["passed_gate"] is True
    assert result.evaluated_lines[0]["is_low_confidence"] is False
    assert result.evaluated_lines[2]["passed_gate"] is True

    # Line 2 should fail due to low confidence (0.22 < 0.35)
    assert result.evaluated_lines[1]["passed_gate"] is False
    assert result.evaluated_lines[1]["is_low_confidence"] is True
    assert "below threshold" in result.evaluated_lines[1]["rejection_reason"]

    # Line 4 should fail due to sanity check (pure symbols)
    assert result.evaluated_lines[3]["passed_gate"] is False
    assert result.evaluated_lines[3]["is_low_confidence"] is True
    assert "Sanity check failed" in result.evaluated_lines[3]["rejection_reason"]

    # Only lines 1 and 3 should be sent to translation
    assert result.text_for_translation == "Hymel vnd ertreich\nmere vnd alle wasser"


def test_whole_image_low_confidence_triggers_fallback():
    """Verify that if all lines fail gating, whole image is flagged and fallback text returned."""
    all_low_lines = [
        {"line_num": 1, "text": "[pmidwngerd", "confidence": 0.03},
        {"line_num": 2, "text": "mtffa snoale", "confidence": 0.01},
        {"line_num": 3, "text": "Fxr }fVno (lExi", "confidence": 0.01},
    ]

    result = gate_ocr_lines_for_translation(all_low_lines, threshold=0.35)

    assert result.can_translate is False
    assert result.whole_image_low_confidence is True
    assert result.text_for_translation == ""
    assert result.notice == FALLBACK_UNRELIABLE_MESSAGE


def test_fast_pipeline_confidence_gate_integration():
    """Verify end-to-end pipeline applies confidence gate and blocks translation for noisy input."""
    manager = FastPipelineManager.get_instance()

    # Generate synthetic noise image (guaranteed low confidence / no recognizable text)
    noise_img = np.random.randint(0, 255, (120, 300, 3), dtype=np.uint8)

    res = manager.process(
        noise_img,
        model_mode="custom",
        confidence_threshold=0.35,
    )

    # Should report unreliable text recognition without translating garbage
    assert res.translated_text == FALLBACK_UNRELIABLE_MESSAGE
    assert res.warning == FALLBACK_UNRELIABLE_MESSAGE


def test_api_predict_configurable_threshold():
    """Verify FastAPI /predict endpoint respects confidence_threshold parameter."""
    client = TestClient(app)

    # Calling with threshold=0.99 forces all lines to gate out
    resp = client.post(
        "/predict",
        data={
            "sample_id": "kannada",
            "model_mode": "custom",
            "confidence_threshold": 0.99,
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # With threshold 0.99, translation must be gated out
    assert data["translated_text"] == FALLBACK_UNRELIABLE_MESSAGE
    assert data["warning"] == FALLBACK_UNRELIABLE_MESSAGE

    # Lines should exist and be marked low confidence
    for line in data["lines"]:
        assert line["is_low_confidence"] is True
        assert line["passed_gate"] is False
