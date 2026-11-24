"""Unified pipeline orchestrator linking Preprocessing, Region Detection, OCR, Quality Gating, LangID, and Translation."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from config import CONFIG
from src.langid import LanguageIDResult, identify_language
from src.ocr.base import OCRResult
from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.quality_gating import QualityCheckResult, evaluate_ocr_quality
from src.ocr.router import OCRRouter
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.trocr_engine import TrOCREngine
from src.preprocessing import TextLineRegion, preprocess_pipeline
from src.translation import TranslationResult, translate_text
from src.utils import format_json_result, get_logger, load_image

LOGGER = get_logger("pipeline")


@dataclass
class PipelineResult:
    """Consolidated output of the end-to-end manuscript processing pipeline."""
    original_image: Optional[np.ndarray]
    processed_image: Optional[np.ndarray]
    line_regions: List[TextLineRegion]
    ocr_result: OCRResult
    quality_check: QualityCheckResult
    langid_result: LanguageIDResult
    translation_result: Optional[TranslationResult]
    translated_image: Optional[np.ndarray] = None
    debug_dir: Optional[str] = None
    export_json: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocr": self.ocr_result.to_dict(),
            "quality": self.quality_check.to_dict(),
            "langid": self.langid_result.to_dict(),
            "translation": self.translation_result.to_dict() if self.translation_result else None,
            "line_count": len(self.line_regions),
            "debug_dir": self.debug_dir,
        }


def save_debug_session(
    original_image: np.ndarray,
    processed_image: np.ndarray,
    line_regions: List[TextLineRegion],
    ocr_result: OCRResult,
    quality_check: QualityCheckResult,
    langid_result: LanguageIDResult,
    diagnostics_extra: Optional[Dict[str, Any]] = None,
    translated_image: Optional[np.ndarray] = None,
) -> str:
    """Save pipeline artifacts and diagnostic report to debug/<timestamp>/ directory."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base_debug_dir = Path(__file__).resolve().parent.parent / "debug" / ts
    base_debug_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save original image
    orig_path = base_debug_dir / "original.png"
    cv2.imwrite(str(orig_path), original_image)

    # 2. Save processed image
    proc_path = base_debug_dir / "processed.png"
    cv2.imwrite(str(proc_path), processed_image)

    # 2b. Save translated image if available
    if translated_image is not None:
        trans_img_path = base_debug_dir / "translated_image.png"
        cv2.imwrite(str(trans_img_path), translated_image)

    # 3. Draw bounding boxes on copy
    annotated = original_image.copy()
    region_records = []

    for idx, reg in enumerate(line_regions or [], 1):
        x1, y1, x2, y2 = reg.bbox_xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Line {idx} ({reg.confidence:.2f})",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        crop_filename = f"crop_{idx:03d}.png"
        crop_path = base_debug_dir / crop_filename
        cv2.imwrite(str(crop_path), reg.crop)

        region_records.append({
            "line_index": idx,
            "bbox_xyxy": list(reg.bbox_xyxy),
            "crop_size_px": [int(reg.crop.shape[1]), int(reg.crop.shape[0])],
            "crop_file": crop_filename,
            "detection_confidence": float(round(reg.confidence, 4)),
            "word_count": int(reg.word_count),
        })

    boxes_path = base_debug_dir / "easyocr_boxes.png"
    cv2.imwrite(str(boxes_path), annotated)

    # 4. Save diagnostics.json
    diag = {
        "timestamp": ts,
        "input_dimensions": [int(original_image.shape[1]), int(original_image.shape[0])],
        "line_count": len(line_regions or []),
        "ocr_result": ocr_result.to_dict(),
        "quality_check": quality_check.to_dict(),
        "langid_result": langid_result.to_dict(),
        "lines": region_records,
        "extra": diagnostics_extra or {},
    }

    diag_path = base_debug_dir / "diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    LOGGER.info("Debug diagnostics saved to: %s", base_debug_dir)
    return str(base_debug_dir)


class ManuscriptPipeline:
    """End-to-end processing pipeline instance."""

    def __init__(
        self,
        router: Optional[OCRRouter] = None,
        easyocr_engine: Optional[EasyOCREngine] = None,
        trocr_engine: Optional[TrOCREngine] = None,
        tesseract_engine: Optional[TesseractEngine] = None,
    ):
        self.easyocr = easyocr_engine or EasyOCREngine()
        self.trocr = trocr_engine or TrOCREngine()
        self.tesseract = tesseract_engine or TesseractEngine()
        self.router = router or OCRRouter(
            easyocr_engine=self.easyocr,
            tesseract_engine=self.tesseract,
            trocr_engine=self.trocr,
        )

    def process_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        mode: str = "Auto",
        target_lang: str = "en",
        deskew: bool = True,
        denoise: bool = True,
        binarize: bool = False,
        debug: bool = False,
        force_translate: bool = False,
        override_lang: Optional[str] = None,
    ) -> PipelineResult:
        """Execute full pipeline on manuscript image."""
        orig_img = load_image(image_input)

        # 1. Preprocess
        processed_img, prep_meta = preprocess_pipeline(
            orig_img,
            deskew=deskew,
            denoise=denoise,
            binarize=binarize,
        )

        # 2. Region Detection & Line Merging (executed on-demand for TrOCR line models or when debug diagnostics enabled)
        line_regions = []
        mode_lower = mode.lower()
        if debug or CONFIG.debug.enabled or any(k in mode_lower for k in ["handwritten", "printed", "trocr"]):
            line_regions = self.easyocr.detect_and_merge_lines(
                processed_img,
                original_image=orig_img,
            )

        # 3. Script-Aware OCR Recognition
        ocr_result = self.router.recognize(
            processed_img,
            mode=mode,
            original_image=orig_img,
            line_regions=line_regions,
        )

        # 4. Quality Gating Assessment
        quality_check = evaluate_ocr_quality(
            text=ocr_result.text,
            confidence=ocr_result.confidence,
            engine_name=ocr_result.engine,
        )

        # 5. Language Identification
        if ocr_result.text:
            langid_res = identify_language(ocr_result.text)
        else:
            langid_res = LanguageIDResult(
                script="Unknown",
                script_confidence=0.0,
                language="Uncertain",
                language_name="Uncertain (No Text)",
                confidence=0.0,
                is_uncertain=True,
            )

        # Apply user language override if specified
        active_src_lang = override_lang if override_lang else langid_res.language

        # 6. Translation Step
        translation_res: Optional[TranslationResult] = None

        # Check if text matches canonical/classical resolvers
        is_classical = False
        from src.classical_languages import (
            resolve_classical_kannada_phrases,
            lookup_canonical_sanskrit,
            resolve_early_modern_german,
            resolve_classical_arabic_phrases,
            resolve_classical_latin_phrases,
        )
        if (
            resolve_classical_kannada_phrases(ocr_result.text)
            or lookup_canonical_sanskrit(ocr_result.text)
            or resolve_early_modern_german(ocr_result.text)
            or resolve_classical_arabic_phrases(ocr_result.text)
            or resolve_classical_latin_phrases(ocr_result.text)
        ):
            is_classical = True

        should_translate = (quality_check.can_translate or force_translate or is_classical) and (active_src_lang != "Uncertain" or is_classical)

        if should_translate and ocr_result.text.strip():
            translation_res = translate_text(
                text=ocr_result.text,
                src_lang=active_src_lang,
                tgt_lang=target_lang,
            )
        elif not quality_check.can_translate and not force_translate:
            translation_res = TranslationResult(
                source_text=ocr_result.text,
                translated_text="",
                src_lang=active_src_lang,
                tgt_lang=target_lang,
                is_skipped=True,
                warning=quality_check.warning or "Translation paused due to low OCR quality.",
            )

        # 6b. Postprocessing: Generate Visual Document Translation (Translated Image)
        translated_img: Optional[np.ndarray] = None
        if translation_res and translation_res.translated_text and line_regions:
            try:
                from src.postprocessing import create_translated_image
                translated_img = create_translated_image(
                    image=orig_img,
                    line_regions=line_regions,
                    translated_text=translation_res.translated_text,
                )
            except Exception as e:
                LOGGER.warning("Visual document translation rendering failed: %s", e)
                translated_img = None

        # 7. Debug Diagnostics
        debug_dir = None
        if debug or CONFIG.debug.enabled:
            debug_dir = save_debug_session(
                original_image=orig_img,
                processed_image=processed_img,
                line_regions=line_regions,
                ocr_result=ocr_result,
                quality_check=quality_check,
                langid_result=langid_res,
                diagnostics_extra={"preprocessing": prep_meta},
                translated_image=translated_img,
            )

        # 8. Standardized JSON Export string
        export_json = format_json_result(
            original_text=ocr_result.text,
            detected_lang=active_src_lang,
            detected_script=langid_res.script,
            lang_confidence=langid_res.confidence,
            translation=translation_res.translated_text if translation_res else "",
            target_lang=target_lang,
            engine_used=ocr_result.engine,
            ocr_confidence=ocr_result.confidence,
            metadata={
                "quality_reliable": quality_check.is_reliable,
                "is_fallback": ocr_result.is_fallback,
                "warnings": [
                    w
                    for w in [ocr_result.warning, quality_check.warning, langid_res.notes]
                    if w
                ],
            },
        )

        return PipelineResult(
            original_image=orig_img,
            processed_image=processed_img,
            line_regions=line_regions or [],
            ocr_result=ocr_result,
            quality_check=quality_check,
            langid_result=langid_res,
            translation_result=translation_res,
            translated_image=translated_img,
            debug_dir=debug_dir,
            export_json=export_json,
        )

    def process_text(
        self,
        text: str,
        target_lang: str = "en",
        override_lang: Optional[str] = None,
    ) -> PipelineResult:
        """Process direct user text input without image OCR."""
        clean_text = text.strip() if text else ""
        langid_res = identify_language(clean_text)
        active_src_lang = override_lang if override_lang else langid_res.language

        quality_check = evaluate_ocr_quality(
            text=clean_text,
            confidence=1.0,
        )

        ocr_res = OCRResult(
            text=clean_text,
            confidence=1.0,
            engine="User Input Text",
        )

        translation_res = None
        if clean_text and active_src_lang != "Uncertain":
            translation_res = translate_text(
                text=clean_text,
                src_lang=active_src_lang,
                tgt_lang=target_lang,
            )

        export_json = format_json_result(
            original_text=clean_text,
            detected_lang=active_src_lang,
            detected_script=langid_res.script,
            lang_confidence=langid_res.confidence,
            translation=translation_res.translated_text if translation_res else "",
            target_lang=target_lang,
            engine_used="User Input Text",
            ocr_confidence=1.0,
        )

        return PipelineResult(
            original_image=None,
            processed_image=None,
            line_regions=[],
            ocr_result=ocr_res,
            quality_check=quality_check,
            langid_result=langid_res,
            translation_result=translation_res,
            export_json=export_json,
        )


DEFAULT_PIPELINE = ManuscriptPipeline()


def process_image_pipeline(
    image_input: Union[str, Path, bytes, np.ndarray],
    mode: str = "Auto",
    target_lang: str = "en",
    deskew: bool = True,
    denoise: bool = True,
    binarize: bool = False,
    debug: bool = False,
    force_translate: bool = False,
    override_lang: Optional[str] = None,
) -> PipelineResult:
    """Functional entrypoint for image pipeline processing."""
    return DEFAULT_PIPELINE.process_image(
        image_input=image_input,
        mode=mode,
        target_lang=target_lang,
        deskew=deskew,
        denoise=denoise,
        binarize=binarize,
        debug=debug,
        force_translate=force_translate,
        override_lang=override_lang,
    )

