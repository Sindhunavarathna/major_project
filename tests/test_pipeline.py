"""Unit tests for pipeline orchestrator, debug diagnostics, and quality gating integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from src.ocr.base import OCRResult
from src.pipeline import ManuscriptPipeline
from src.preprocessing import TextLineRegion


def test_process_text():
    pipeline = ManuscriptPipeline()
    with patch("src.pipeline.translate_text") as mock_trans:
        mock_trans.return_value = MagicMock(
            translated_text="अनुवादित पाठ",
            is_skipped=False,
            is_pivot=False,
            warning=None,
        )
        res = pipeline.process_text(
            "Ancient manuscripts preserve historical records.",
            target_lang="hi",
        )
        assert res.ocr_result.text == "Ancient manuscripts preserve historical records."
        assert res.langid_result.language == "en"
        assert res.quality_check.is_reliable is True
        assert res.export_json is not None
        assert res.translation_result.translated_text == "अनुवादित पाठ"


def test_pipeline_junk_symbol_triggers_quality_warning_and_blocks_translation(synthetic_text_image: np.ndarray):
    pipeline = ManuscriptPipeline()

    mock_ocr = OCRResult(text="# 0", confidence=0.25, engine="MockJunk")
    with patch.object(pipeline.easyocr, "detect_and_merge_lines", return_value=[]), \
         patch.object(pipeline.router, "recognize", return_value=mock_ocr):
        res = pipeline.process_image(
            synthetic_text_image,
            mode="Auto",
            target_lang="en",
            force_translate=False,
        )
        assert res.quality_check.is_reliable is False
        assert res.quality_check.can_translate is False
        # Verify auto-translation was skipped and not hallucinated
        assert res.translation_result.is_skipped is True
        assert res.translation_result.translated_text == ""


def test_pipeline_debug_session_creation(synthetic_text_image: np.ndarray, tmp_path: Path):
    pipeline = ManuscriptPipeline()
    mock_ocr = OCRResult(
        text="Valid manuscript text for testing",
        confidence=0.92,
        engine="MockEngine",
    )
    mock_region = TextLineRegion(
        bbox_xyxy=(10, 10, 100, 40),
        confidence=0.95,
        word_count=2,
        crop=synthetic_text_image[10:40, 10:100],
    )

    with patch.object(pipeline.easyocr, "detect_and_merge_lines", return_value=[mock_region]), \
         patch.object(pipeline.router, "recognize", return_value=mock_ocr), \
         patch("src.pipeline.translate_text") as mock_trans:
        mock_trans.return_value = MagicMock(
            translated_text="Translated text",
            is_skipped=False,
            is_pivot=False,
            warning=None,
        )

        res = pipeline.process_image(
            synthetic_text_image,
            mode="Auto",
            target_lang="en",
            debug=True,
        )

        assert res.debug_dir is not None
        debug_p = Path(res.debug_dir)
        assert debug_p.exists()
        assert (debug_p / "original.png").exists()
        assert (debug_p / "processed.png").exists()
        assert (debug_p / "easyocr_boxes.png").exists()
        assert (debug_p / "crop_001.png").exists()
        assert (debug_p / "diagnostics.json").exists()
