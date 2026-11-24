"""Tesseract OCR engine wrapper with language mapping and graceful tessdata fallback."""

from __future__ import annotations

import os
from typing import Any, List, Optional
import numpy as np
import pytesseract
from pytesseract import Output

from config import CONFIG
from src.ocr.base import BaseOCREngine, OCRResult
from src.utils import get_logger

LOGGER = get_logger("tesseract_engine")


class TesseractEngine(BaseOCREngine):
    """PyTesseract engine wrapper supporting confidence estimation and layout analysis."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        super().__init__(name="Tesseract")
        cmd = tesseract_cmd or CONFIG.ocr.tesseract.tesseract_cmd
        if os.path.exists(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd
        self._available_langs: Optional[List[str]] = None

    @property
    def available_languages(self) -> List[str]:
        if self._available_langs is None:
            try:
                self._available_langs = pytesseract.get_languages()
            except Exception as e:
                LOGGER.warning("Could not query Tesseract languages: %s", e)
                self._available_langs = ["eng"]
        return self._available_langs

    def _map_language(self, lang: Optional[str]) -> str:
        """Map generic language codes to Tesseract 3-letter codes."""
        mapping = {
            "en": "eng",
            "hi": "hin",
            "sa": "san",
            "kn": "kan",
            "ta": "tam",
            "te": "tel",
        }
        if not lang:
            return CONFIG.ocr.tesseract.default_lang
        return mapping.get(lang.lower(), lang)

    def recognize(
        self,
        image: np.ndarray,
        lang: Optional[str] = None,
        psm: Optional[int] = None,
        oem: Optional[int] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Run Tesseract recognition on image array."""
        target_lang = self._map_language(lang)
        is_fallback = False
        warning = None

        # Verify language is installed in tessdata; fallback if missing
        if target_lang not in self.available_languages:
            warning = (
                f"Tesseract traineddata for '{target_lang}' is not installed in tessdata "
                f"(available: {self.available_languages}). Falling back to 'eng'."
            )
            LOGGER.warning(warning)
            target_lang = "eng"
            is_fallback = True

        oem_val = oem if oem is not None else CONFIG.ocr.tesseract.oem
        psm_val = psm if psm is not None else CONFIG.ocr.tesseract.psm
        config_str = f"--oem {oem_val} --psm {psm_val}"

        try:
            # Use image_to_data to extract words and confidence scores
            data = pytesseract.image_to_data(
                image,
                lang=target_lang,
                config=config_str,
                output_type=Output.DICT,
            )

            words: List[str] = []
            confidences: List[float] = []

            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text_item = data["text"][i].strip()
                conf_item = float(data["conf"][i])

                # Tesseract reports -1 for empty blocks/lines
                if text_item and conf_item >= 0:
                    words.append(text_item)
                    confidences.append(conf_item / 100.0)

            full_text = " ".join(words)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            return OCRResult(
                text=full_text,
                confidence=round(avg_conf, 4),
                engine=self.name,
                is_fallback=is_fallback,
                warning=warning,
                raw_details={
                    "language_used": target_lang,
                    "word_count": len(words),
                    "confidences": confidences,
                },
            )

        except Exception as exc:
            LOGGER.error("Tesseract recognition error: %s", exc, exc_info=True)
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                is_fallback=is_fallback,
                warning=f"Tesseract failure: {str(exc)}",
                raw_details={"error": str(exc)},
            )

