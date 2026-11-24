"""Unit tests for OCR engine wrappers, script-aware router, and quality gating."""

from unittest.mock import MagicMock
import numpy as np

from src.ocr.base import OCRResult
from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.quality_gating import evaluate_ocr_quality
from src.ocr.router import OCRRouter
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.trocr_engine import TrOCREngine


def test_ocr_result_serialization():
    res = OCRResult(
        text="Sample text",
        confidence=0.92,
        engine="TestEngine",
        is_fallback=True,
        warning="Test warning",
        raw_details={"key": "val"},
    )
    d = res.to_dict()
    assert d["text"] == "Sample text"
    assert d["confidence"] == 0.92
    assert d["engine"] == "TestEngine"
    assert d["is_fallback"] is True
    assert d["warning"] == "Test warning"


def test_quality_gating_junk_detection():
    # Test "# 0" symbol noise is flagged as unreliable
    res_junk = evaluate_ocr_quality(text="# 0", confidence=0.25)
    assert res_junk.is_reliable is False
    assert res_junk.can_translate is False
    assert "symbols" in res_junk.warning.lower() or "ratio" in res_junk.warning.lower()

    # Test valid text passes
    res_valid = evaluate_ocr_quality(
        text="Ancient manuscripts preserve knowledge",
        confidence=0.88,
    )
    assert res_valid.is_reliable is True
    assert res_valid.can_translate is True
    assert res_valid.warning is None


def test_tesseract_engine_english(synthetic_text_image: np.ndarray):
    engine = TesseractEngine()
    result = engine.recognize(synthetic_text_image, lang="en")
    assert isinstance(result, OCRResult)
    assert result.engine == "Tesseract"
    assert len(result.text) > 0
    assert "ANCIENT" in result.text.upper() or "MANUSCRIPT" in result.text.upper()
    assert result.confidence > 0.0


def test_tesseract_engine_fallback_for_missing_lang(synthetic_text_image: np.ndarray):
    engine = TesseractEngine()
    result = engine.recognize(synthetic_text_image, lang="san")
    assert isinstance(result, OCRResult)
    assert result.is_fallback is True
    assert result.warning is not None
    assert "not installed in tessdata" in result.warning


def test_trocr_engine_missing_sanskrit_model(synthetic_text_image: np.ndarray):
    engine = TrOCREngine(sanskrit_model_path="non/existent/models/sanskrit_trocr")
    mod, proc, target, is_fallback, warning = engine.get_model_and_processor("sanskrit")
    assert mod is None
    assert is_fallback is True
    assert "Fine-tuned Sanskrit TrOCR model not found" in warning


def test_ocr_router_devanagari_routes_cleanly(synthetic_text_image: np.ndarray):
    mock_easy = MagicMock(spec=EasyOCREngine)
    mock_easy.name = "EasyOCR"
    mock_easy.detect_and_merge_lines.return_value = []
    mock_easy.recognize.return_value = OCRResult(
        text="यह एक नमूना है",
        confidence=0.85,
        engine="EasyOCR",
    )

    mock_trocr = MagicMock(spec=TrOCREngine)
    mock_trocr.name = "TrOCR"

    router = OCRRouter(easyocr_engine=mock_easy, trocr_engine=mock_trocr)
    res = router.recognize(synthetic_text_image, mode="Sanskrit/Devanagari")
    assert "Fallback: EasyOCR" in res.engine or "TrOCR" in res.engine
