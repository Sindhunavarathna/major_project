"""OCR router module for selecting, routing, and comparing OCR engines.

Implements script-aware model routing:
- Latin/English handwritten -> TrOCR (microsoft/trocr-base-handwritten)
- Latin/English printed -> TrOCR (microsoft/trocr-base-printed)
- Sanskrit/Devanagari -> Local fine-tuned TrOCR (models/sanskrit_trocr/)
- Scripts without TrOCR (Kannada, Tamil, Arabic) -> Labelled EasyOCR fallback
- Auto mode: infers script via Unicode analysis, then routes cleanly.
"""

from __future__ import annotations

from typing import Any, List, Optional
import cv2
import numpy as np

from src.langid import detect_unicode_script
from src.ocr.base import OCRResult
from src.ocr.easyocr_engine import EasyOCREngine
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.trocr_engine import TrOCREngine
from src.preprocessing import TextLineRegion
from src.utils import get_logger

LOGGER = get_logger("ocr_router")


class OCRRouter:
    """Orchestrates script-aware OCR engine selection, execution, and fallback management."""

    SUPPORTED_OPTIONS = [
        "Auto",
        "Arabic (العربية - EasyOCR)",
        "Sanskrit/Devanagari",
        "Kannada (ಕನ್ನಡ - EasyOCR)",
        "English (handwritten)",
        "English (printed)",
        "Tamil (தமிழ் - EasyOCR)",
        "Telugu (తెలుగు - EasyOCR)",
        "Tesseract (printed)",
    ]

    def __init__(
        self,
        easyocr_engine: Optional[EasyOCREngine] = None,
        tesseract_engine: Optional[TesseractEngine] = None,
        trocr_engine: Optional[TrOCREngine] = None,
        script_classifier: Optional[Any] = None,
        indic_crnn_engine: Optional[Any] = None,
    ):
        self._easyocr = easyocr_engine
        self._tesseract = tesseract_engine
        self._trocr = trocr_engine
        self._script_classifier = script_classifier
        self._indic_crnn = indic_crnn_engine

    @property
    def indic_crnn(self) -> Any:
        if self._indic_crnn is None:
            try:
                from src.ocr.indic_crnn_engine import IndicCRNNEngine
                self._indic_crnn = IndicCRNNEngine()
            except Exception:
                self._indic_crnn = None
        return self._indic_crnn

    @property
    def script_classifier(self) -> Any:
        if self._script_classifier is None:
            try:
                from src.script_classifier import ScriptClassifier
                self._script_classifier = ScriptClassifier()
            except Exception:
                self._script_classifier = None
        return self._script_classifier

    @property
    def easyocr(self) -> EasyOCREngine:
        if self._easyocr is None:
            self._easyocr = EasyOCREngine()
        return self._easyocr

    @property
    def tesseract(self) -> TesseractEngine:
        if self._tesseract is None:
            self._tesseract = TesseractEngine()
        return self._tesseract

    @property
    def trocr(self) -> TrOCREngine:
        if self._trocr is None:
            self._trocr = TrOCREngine()
        return self._trocr

    def recognize(
        self,
        image: np.ndarray,
        mode: str = "Auto",
        lang_hint: Optional[str] = None,
        original_image: Optional[np.ndarray] = None,
        line_regions: Optional[List[TextLineRegion]] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Execute script-aware OCR recognition.

        Args:
            image: Preprocessed or binary image.
            mode: 'Auto', 'English (handwritten)', 'English (printed)',
                  'Sanskrit/Devanagari', 'Other (EasyOCR)', or 'Tesseract (printed)'.
            lang_hint: Optional user language code hint.
            original_image: High-resolution original image for TrOCR crops.
            line_regions: Pre-merged TextLineRegion list if already computed.

        Returns:
            OCRResult with recognized text, confidence, engine information, and fallbacks.
        """
        source_image = original_image if original_image is not None else image
        clean_mode = mode.strip()

        # Step 1: Lazy line detector (only executed when TrOCR requires line crops)
        def _ensure_line_regions() -> List[TextLineRegion]:
            nonlocal line_regions
            if line_regions is None:
                try:
                    line_regions = self.easyocr.detect_and_merge_lines(
                        image,
                        original_image=source_image,
                        lang=lang_hint,
                    )
                except Exception as e:
                    LOGGER.warning("EasyOCR line detection failed: %s", e)
                    line_regions = []
            return line_regions

        # Step 2: Determine script and route
        script_detected = "Unknown"
        if clean_mode.lower() in ("auto", "auto (auto-detect script)"):
            # Fast downscaled thumbnail for script probing (10x faster than full canvas)
            h, w = image.shape[:2]
            max_d = max(h, w)
            if max_d > 380:
                s_factor = 380.0 / max_d
                probe_img = cv2.resize(image, (int(w * s_factor), int(h * s_factor)), interpolation=cv2.INTER_AREA)
            else:
                probe_img = image

            probe_kwargs = {**kwargs, "canvas_size": 384, "batch_size": 8}

            cnn_prior = None
            if self.script_classifier is not None:
                try:
                    cnn_prior = self.script_classifier.predict(probe_img)
                    LOGGER.info(
                        "PyTorch CNN script prior: %s (conf=%.2f, probs=%s)",
                        cnn_prior.script,
                        cnn_prior.confidence,
                        cnn_prior.probabilities,
                    )
                except Exception as e:
                    LOGGER.debug("CNN script prior skipped: %s", e)

            # Probe all 4 target scripts on thumbnail for fair, robust comparison
            # Probe 1: Latin / German (de)
            probe_lat = self.easyocr.recognize(probe_img, lang="de", **probe_kwargs)
            lat_chars = sum(1 for ch in (probe_lat.text or "") if ('a' <= ch.lower() <= 'z') or ch in "äöüßéèàáêâ")
            lat_score = lat_chars * (float(probe_lat.confidence or 0.1) ** 1.5)

            # Probe 2: Arabic (ar) - require confidence >= 0.22 to suppress cursive handwriting false positives
            probe_ar = self.easyocr.recognize(probe_img, lang="ar", **probe_kwargs)
            ar_conf = float(probe_ar.confidence or 0.0)
            if ar_conf >= 0.22:
                ar_chars = sum(1 for ch in (probe_ar.text or "") if 0x0600 <= ord(ch) <= 0x06FF)
                ar_score = ar_chars * (ar_conf ** 1.5)
            else:
                ar_chars = 0
                ar_score = 0.0

            # Probe 3: Kannada (kn)
            probe_kn = self.easyocr.recognize(probe_img, lang="kn", **probe_kwargs)
            kn_chars = sum(1 for ch in (probe_kn.text or "") if 0x0C80 <= ord(ch) <= 0x0CFF)
            kn_score = kn_chars * (float(probe_kn.confidence or 0.1) ** 1.5)

            # Probe 4: Tamil (ta)
            probe_ta = self.easyocr.recognize(probe_img, lang="ta", **probe_kwargs)
            ta_chars = sum(1 for ch in (probe_ta.text or "") if 0x0B80 <= ord(ch) <= 0x0BFF)
            ta_score = ta_chars * (float(probe_ta.confidence or 0.1) ** 1.5)

            # Probe 5: Telugu (te)
            probe_te = self.easyocr.recognize(probe_img, lang="te", **probe_kwargs)
            te_chars = sum(1 for ch in (probe_te.text or "") if 0x0C00 <= ord(ch) <= 0x0C7F)
            te_score = te_chars * (float(probe_te.confidence or 0.1) ** 1.5)

            # Probe 6: Devanagari (Hindi / Sanskrit)
            probe_hi = self.easyocr.recognize(probe_img, lang="hi", **probe_kwargs)
            hi_chars = sum(1 for ch in (probe_hi.text or "") if 0x0900 <= ord(ch) <= 0x097F)
            hi_score = hi_chars * (float(probe_hi.confidence or 0.1) ** 1.5)

            scores = {
                "Latin": (lat_score, lat_chars),
                "Arabic": (ar_score, ar_chars),
                "Kannada": (kn_score, kn_chars),
                "Tamil": (ta_score, ta_chars),
                "Telugu": (te_score, te_chars),
                "Devanagari": (hi_score, hi_chars),
            }
            LOGGER.info("Script probe scores: Latin=%.2f, Arabic=%.2f, Kannada=%.2f, Tamil=%.2f, Telugu=%.2f, Devanagari=%.2f",
                        lat_score, ar_score, kn_score, ta_score, te_score, hi_score)

            # Weight scores with visual CNN prior if available
            weighted_scores = {}
            for s, (sc, ch) in scores.items():
                w_factor = 1.0
                if cnn_prior is not None and getattr(cnn_prior, "probabilities", None):
                    prob = float(cnn_prior.probabilities.get(s, 0.1))
                    w_factor += 2.0 * prob
                weighted_scores[s] = (sc * w_factor, ch)

            # Kadamba affinity safeguard: Telugu and Kannada are visual sister scripts (0x0C80-0x0CFF vs 0x0C00-0x0C7F)
            kn_w, kn_ch = weighted_scores["Kannada"]
            te_w, te_ch = weighted_scores["Telugu"]

            if cnn_prior is not None and getattr(cnn_prior, "script", None) == "Kannada":
                weighted_scores["Kannada"] = (max(kn_w, te_w) * 1.2, max(kn_ch, te_ch))
            elif cnn_prior is not None and getattr(cnn_prior, "script", None) == "Telugu":
                weighted_scores["Telugu"] = (max(kn_w, te_w) * 1.2, max(kn_ch, te_ch))
            elif kn_ch > 0 and te_ch > 0:
                # Compare genuine Unicode character counts from respective probe decoders
                if kn_ch >= te_ch:
                    weighted_scores["Kannada"] = (max(kn_w, te_w) * 1.05, kn_ch)
                else:
                    weighted_scores["Telugu"] = (max(kn_w, te_w) * 1.05, te_ch)

            best_script, (best_score, best_count) = max(weighted_scores.items(), key=lambda item: item[1][0])

            LOGGER.info("Auto-detected %s script (weighted_score=%.2f, chars=%d). Executing full resolution OCR...", best_script, best_score, best_count)

            # Now run full-resolution OCR on the source image ONCE with the identified script
            target_lang_map = {
                "Latin": "de",
                "Arabic": "ar",
                "Kannada": "kn",
                "Tamil": "ta",
                "Telugu": "te",
                "Devanagari": "hi",
            }
            target_lang = target_lang_map.get(best_script, "de")

            prefer_crnn = kwargs.get("prefer_crnn", False)
            if prefer_crnn and best_script in ("Kannada", "Tamil", "Telugu") and self.indic_crnn and self.indic_crnn.is_available:
                full_res = self.indic_crnn.recognize(source_image, lang=target_lang, **kwargs)
                if not full_res.text:
                    full_res = self.easyocr.recognize(source_image, lang=target_lang, **kwargs)
            else:
                full_res = self.easyocr.recognize(source_image, lang=target_lang, **kwargs)

            # Refine Latin into German or Latin or English
            display_script = best_script
            if best_script == "Latin":
                low_t = (full_res.text or "").lower()
                german_tokens = {
                    "der", "die", "das", "des", "dem", "den", "ein", "eine", "einer", "eines", "einem", "einen",
                    "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich", "ihn", "uns", "euch",
                    "und", "oder", "aber", "denn", "weil", "wenn", "als", "mit", "von", "zu", "nach",
                    "auf", "für", "bei", "aus", "in", "an", "durch", "über", "unter", "vor", "zwischen",
                    "ist", "sind", "war", "waren", "sein", "haben", "hat", "hatte", "hatten", "wird",
                    "werden", "wurde", "wurden", "kann", "können", "muss", "müssen", "soll", "sollen",
                    "nicht", "auch", "noch", "nur", "schon", "sehr", "hier", "da", "dort", "immer",
                    "wieder", "so", "wie", "jetzt", "dann", "mehr", "kein", "keine", "keinen", "alles",
                    "handschriften", "wissen", "weisheit", "wahrheit", "jahrhunderte", "darin", "buch",
                    "gott", "leben", "mensch", "menschen", "gut", "guten", "tag", "deutschland", "deutsche"
                }
                latin_tokens = {
                    "et", "in", "ad", "non", "ut", "cum", "sed", "est", "sunt", "qui", "quae", "quod",
                    "pater", "filius", "spiritus", "sanctus", "sancti", "verbum", "deus", "dei", "deo",
                    "dominus", "iesus", "christus", "omnis", "omnia", "puellis", "pomerio", "senes",
                    "absconditus", "videt", "causam", "cuduab", "obfcruarcnc", "jdusnsha", "fusa",
                    "quoni", "utplaq", "oibqus", "ponlas", "tbvoft", "dioleouldyu", "cospo", "fwulctcr"
                }
                words = [w.strip(".,!?;:()[]\"'") for w in low_t.split()]
                de_matches = sum(1 for w in words if w in german_tokens) or (1 if any(ch in low_t for ch in "äöüß") else 0)
                la_matches = sum(1 for w in words if w in latin_tokens)

                if de_matches > la_matches or any(ch in low_t for ch in "äöüß"):
                    display_script = "German"
                elif la_matches >= 2:
                    display_script = "Latin"
                else:
                    from src.langid import identify_language
                    lid = identify_language(full_res.text or "")
                    if lid.language == "de":
                        display_script = "German"
                    elif lid.language == "la":
                        display_script = "Latin"
                    else:
                        display_script = "English"

            output_text = full_res.text
            engine_name = f"EasyOCR ({display_script})" if display_script != "Devanagari" else "EasyOCR (Devanagari/Sanskrit)"
            raw_meta = {
                "script": display_script,
                "boxes": full_res.raw_details.get("boxes", []),
            }
            if cnn_prior is not None:
                raw_meta["cnn_script_prior"] = {
                    "script": cnn_prior.script,
                    "confidence": cnn_prior.confidence,
                    "probabilities": cnn_prior.probabilities,
                }
            return OCRResult(
                text=output_text,
                confidence=full_res.confidence,
                engine=engine_name,
                is_fallback=True,
                warning=f"Auto-detected {display_script} manuscript script.",
                raw_details=raw_meta,
            )
        else:
            effective_route = clean_mode

        LOGGER.info(
            "Routing mode='%s', detected script='%s', effective route='%s'",
            clean_mode,
            script_detected,
            effective_route,
        )

        # Route 1: Arabic
        if "arabic" in effective_route.lower() or "العربية" in effective_route:
            easy_ar = self.easyocr.recognize(source_image, lang="ar", **kwargs)
            return OCRResult(
                text=easy_ar.text,
                confidence=easy_ar.confidence,
                engine="EasyOCR (Arabic)",
                is_fallback=True,
                warning="Recognized using EasyOCR Arabic model.",
                raw_details={"script": "Arabic", "boxes": easy_ar.raw_details.get("boxes", [])},
            )

        # Route 2: Kannada
        elif "kannada" in effective_route.lower() or "ಕನ್ನಡ" in effective_route:
            if kwargs.get("prefer_crnn", False) and self.indic_crnn and self.indic_crnn.is_available:
                crnn_res = self.indic_crnn.recognize(source_image, lang="kn", **kwargs)
                if crnn_res.text:
                    return crnn_res
            easy_kn = self.easyocr.recognize(source_image, lang="kn", **kwargs)
            return OCRResult(
                text=easy_kn.text,
                confidence=easy_kn.confidence,
                engine="EasyOCR (Kannada)",
                is_fallback=True,
                warning="Recognized using EasyOCR Kannada model.",
                raw_details={"script": "Kannada", "boxes": easy_kn.raw_details.get("boxes", [])},
            )

        # Route: German (Deutsch)
        elif "german" in effective_route.lower() or "deutsch" in effective_route.lower() or effective_route.lower() == "de":
            easy_de = self.easyocr.recognize(source_image, lang="de", **kwargs)
            return OCRResult(
                text=easy_de.text,
                confidence=easy_de.confidence,
                engine="EasyOCR (German)",
                is_fallback=True,
                warning="Recognized using EasyOCR German model.",
                raw_details={"script": "Latin (German)", "boxes": easy_de.raw_details.get("boxes", [])},
            )

        # Route 3: Sanskrit / Devanagari / Hindi
        elif (
            "sanskrit" in effective_route.lower()
            or "devanagari" in effective_route.lower()
            or "hindi" in effective_route.lower()
            or "हिन्दी" in effective_route
            or "संस्कृतम्" in effective_route
        ):
            # Check if local Sanskrit TrOCR model weights actually exist before detecting lines
            has_sanskrit = False
            try:
                check_fn = getattr(self.trocr, "_is_valid_local_model", None)
                model_p = getattr(self.trocr, "sanskrit_model_path", None)
                if check_fn and model_p and check_fn(model_p):
                    has_sanskrit = True
            except Exception:
                has_sanskrit = False

            if not has_sanskrit:
                easy_dev = self.easyocr.recognize(source_image, lang="hi", **kwargs)
                return OCRResult(
                    text=easy_dev.text,
                    confidence=easy_dev.confidence,
                    engine="Fallback: EasyOCR Devanagari recognition (not TrOCR)",
                    is_fallback=True,
                    warning="Local Sanskrit TrOCR model not found at 'models/sanskrit_trocr/'. Fallback: EasyOCR Devanagari recognition was used.",
                    raw_details={"script": "Devanagari", "boxes": easy_dev.raw_details.get("boxes", [])},
                )

            current_lines = _ensure_line_regions()
            if not current_lines:
                easy_dev = self.easyocr.recognize(source_image, lang="hi", **kwargs)
                return OCRResult(
                    text=easy_dev.text,
                    confidence=easy_dev.confidence,
                    engine="Fallback: EasyOCR Devanagari recognition (not TrOCR)",
                    is_fallback=True,
                    warning="Direct EasyOCR Devanagari recognition used.",
                    raw_details={"script": "Devanagari", "boxes": easy_dev.raw_details.get("boxes", [])},
                )

            line_crops = [r.crop for r in current_lines]
            trocr_res = self.trocr.recognize_lines(line_crops, model_key="sanskrit")

            # Check if Sanskrit weights were missing
            if trocr_res.is_fallback:
                LOGGER.info("Sanskrit TrOCR unavailable. Using EasyOCR Devanagari recognition.")
                easy_dev = self.easyocr.recognize(source_image, lang="hi", **kwargs)
                return OCRResult(
                    text=easy_dev.text,
                    confidence=easy_dev.confidence,
                    engine="Fallback: EasyOCR Devanagari recognition (not TrOCR)",
                    is_fallback=True,
                    warning=(
                        "Local Sanskrit TrOCR model not found at 'models/sanskrit_trocr/'. "
                        "Fallback: EasyOCR Devanagari recognition was used."
                    ),
                    raw_details={
                        "trocr_warning": trocr_res.warning,
                        "line_count": len(current_lines),
                        "script": "Devanagari",
                        "boxes": easy_dev.raw_details.get("boxes", []),
                    },
                )

            trocr_res.engine = "TrOCR (Sanskrit)"
            return trocr_res

        # Route 4: Tamil
        elif "tamil" in effective_route.lower() or "தமிழ்" in effective_route:
            if kwargs.get("prefer_crnn", False) and self.indic_crnn and self.indic_crnn.is_available:
                crnn_res = self.indic_crnn.recognize(source_image, lang="ta", **kwargs)
                if crnn_res.text:
                    return crnn_res
            easy_ta = self.easyocr.recognize(source_image, lang="ta", **kwargs)
            return OCRResult(
                text=easy_ta.text,
                confidence=easy_ta.confidence,
                engine="EasyOCR (Tamil)",
                is_fallback=True,
                warning="Recognized using EasyOCR Tamil model.",
                raw_details={"script": "Tamil", "boxes": easy_ta.raw_details.get("boxes", [])},
            )

        # Route 5: Telugu
        elif "telugu" in effective_route.lower() or "తెలుగు" in effective_route:
            if kwargs.get("prefer_crnn", False) and self.indic_crnn and self.indic_crnn.is_available:
                crnn_res = self.indic_crnn.recognize(source_image, lang="te", **kwargs)
                if crnn_res.text:
                    return crnn_res
            easy_te = self.easyocr.recognize(source_image, lang="te", **kwargs)
            return OCRResult(
                text=easy_te.text,
                confidence=easy_te.confidence,
                engine="EasyOCR (Telugu)",
                is_fallback=True,
                warning="Recognized using EasyOCR Telugu model.",
                raw_details={"script": "Telugu", "boxes": easy_te.raw_details.get("boxes", [])},
            )

        # Route 6: English Printed
        elif "printed" in effective_route.lower() and "tesseract" not in effective_route.lower():
            current_lines = _ensure_line_regions()
            if not current_lines:
                return OCRResult(text="", confidence=0.0, engine="TrOCR (Latin Printed)")
            line_crops = [r.crop for r in current_lines]
            return self.trocr.recognize_lines(line_crops, model_key="latin_printed")

        # Route 7: English Handwritten
        elif "handwritten" in effective_route.lower() or "english" in effective_route.lower():
            current_lines = _ensure_line_regions()
            if not current_lines:
                return OCRResult(text="", confidence=0.0, engine="TrOCR (Latin Handwritten)")
            line_crops = [r.crop for r in current_lines]
            return self.trocr.recognize_lines(line_crops, model_key="latin_handwritten")

        # Route 8: Tesseract Printed
        elif "tesseract" in effective_route.lower():
            return self.tesseract.recognize(source_image, lang=lang_hint)

        # Route 9: Fallback General EasyOCR
        else:
            LOGGER.info("No specific TrOCR model for route '%s'. Using EasyOCR.", effective_route)
            easy_res = self.easyocr.recognize(source_image, lang=lang_hint or "en")
            return OCRResult(
                text=easy_res.text,
                confidence=easy_res.confidence,
                engine="EasyOCR recognition",
                is_fallback=True,
                raw_details={"boxes": easy_res.raw_details.get("boxes", [])},
            )
