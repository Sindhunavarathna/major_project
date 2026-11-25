"""
High-Performance Pipeline Manager for Multilingual Handwritten Note Recognition & Translation.

Optimizations:
1. One-time server startup model pre-loading (all models kept in memory).
2. Distilled NLLB-200 (600M) with PyTorch int8 dynamic quantization on CPU.
3. Aspect-ratio preserving downscaling for high-resolution manuscript scans (>1280px).
4. Batched tensor evaluation for multi-line Indic CRNN forward passes.
5. High-resolution wall-clock profiling per stage (preprocess, classifier, OCR, langid, translation).
6. Comprehensive input language detection with confidence gating and "Uncertain" handling.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from src.language_id import distinguish_german_vs_english
from src.ocr.base import OCRResult
from src.preprocessing import (
    AblationConfig,
    TextLineRegion,
    enhance_contrast_mild,
    run_ablation_pipeline,
    sauvola_threshold,
)
from src.utils import get_logger, load_image

LOGGER = get_logger("fast_pipeline")

# Language metadata dictionary
LANGUAGE_METADATA = {
    "kannada": {
        "name": "Kannada",
        "native_name": "ಕನ್ನಡ",
        "code": "kn",
        "script": "Kannada",
        "nllb_code": "kan_Knda",
    },
    "tamil": {
        "name": "Tamil",
        "native_name": "தமிழ்",
        "code": "ta",
        "script": "Tamil",
        "nllb_code": "tam_Taml",
    },
    "telugu": {
        "name": "Telugu",
        "native_name": "తెలుగు",
        "code": "te",
        "script": "Telugu",
        "nllb_code": "tel_Telu",
    },
    "german": {
        "name": "German",
        "native_name": "Deutsch",
        "code": "de",
        "script": "Latin",
        "nllb_code": "deu_Latn",
    },
    "english": {
        "name": "English",
        "native_name": "English",
        "code": "en",
        "script": "Latin",
        "nllb_code": "eng_Latn",
    },
    "arabic": {
        "name": "Arabic",
        "native_name": "العربية",
        "code": "ar",
        "script": "Arabic",
        "nllb_code": "arb_Arab",
    },
    "sanskrit": {
        "name": "Sanskrit",
        "native_name": "संस्कृतम्",
        "code": "sa",
        "script": "Devanagari",
        "nllb_code": "san_Deva",
    },
    "devanagari": {
        "name": "Sanskrit / Hindi",
        "native_name": "संस्कृतम् / हिन्दी",
        "code": "sa",
        "script": "Devanagari",
        "nllb_code": "san_Deva",
    },
    "hindi": {
        "name": "Hindi",
        "native_name": "हिन्दी",
        "code": "hi",
        "script": "Devanagari",
        "nllb_code": "hin_Deva",
    },
}


@dataclass
class StageTiming:
    """Per-stage execution wall-clock time in milliseconds."""
    preprocess_ms: float = 0.0
    script_classifier_ms: float = 0.0
    ocr_ms: float = 0.0
    language_id_ms: float = 0.0
    translation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "preprocess_ms": round(self.preprocess_ms, 2),
            "script_classifier_ms": round(self.script_classifier_ms, 2),
            "ocr_ms": round(self.ocr_ms, 2),
            "language_id_ms": round(self.language_id_ms, 2),
            "translation_ms": round(self.translation_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class FastPipelineResult:
    """Comprehensive structured result of the fast inference pipeline."""
    success: bool
    detected_language: str
    detected_language_code: str
    detected_script: str
    language_native_name: str
    confidence: float
    is_uncertain: bool
    confidence_tier: str  # 'high', 'medium', 'uncertain'
    recognized_text: str
    translated_text: str
    engine_used: str
    timing: StageTiming
    lines: List[Dict[str, Any]] = field(default_factory=list)
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "detected_language": self.detected_language,
            "detected_language_code": self.detected_language_code,
            "detected_script": self.detected_script,
            "language_native_name": self.language_native_name,
            "confidence": round(self.confidence, 4),
            "is_uncertain": self.is_uncertain,
            "confidence_tier": self.confidence_tier,
            "recognized_text": self.recognized_text,
            "translated_text": self.translated_text,
            "engine_used": self.engine_used,
            "timing": self.timing.to_dict(),
            "lines": self.lines,
            "warning": self.warning,
        }


class FastPipelineManager:
    """Singleton Manager pre-loading and holding all models in memory."""

    _instance: Optional[FastPipelineManager] = None

    @classmethod
    def get_instance(cls) -> FastPipelineManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        LOGGER.info("Initializing FastPipelineManager: Pre-loading all models into memory...")
        t_start = time.perf_counter()

        # 1. Device configuration
        self.device = torch.device("cpu")

        # 2. PyTorch Visual Script Classifier (5-class CNN)
        self.script_classifier = None
        try:
            from src.script_classifier import ScriptClassifier
            self.script_classifier = ScriptClassifier(device=str(self.device))
            LOGGER.info("Loaded ScriptClassifier (CNN).")
        except Exception as e:
            LOGGER.warning("Could not pre-load ScriptClassifier: %s", e)

        # 3. Shared Indic CRNN Engine (Kannada, Tamil, Telugu)
        self.indic_crnn = None
        try:
            from src.ocr.indic_crnn_engine import IndicCRNNEngine
            self.indic_crnn = IndicCRNNEngine(device=str(self.device))
            LOGGER.info("Loaded IndicCRNNEngine (CRNN + CTC).")
        except Exception as e:
            LOGGER.warning("Could not pre-load IndicCRNNEngine: %s", e)

        # 4. TrOCR / EasyOCR engines
        self.easyocr = None
        try:
            from src.ocr.easyocr_engine import EasyOCREngine
            self.easyocr = EasyOCREngine(gpu=False)
            LOGGER.info("Loaded EasyOCREngine.")
        except Exception as e:
            LOGGER.warning("Could not pre-load EasyOCREngine: %s", e)

        self.trocr = None
        try:
            from src.ocr.trocr_engine import TrOCREngine
            self.trocr = TrOCREngine(device=str(self.device))
            LOGGER.info("Loaded TrOCREngine.")
        except Exception as e:
            LOGGER.warning("Could not pre-load TrOCREngine: %s", e)

        # 5. Distilled NLLB-200 with dynamic int8 quantization on CPU
        self.translator = None
        try:
            from src.translator import NLLBTranslator
            self.translator = NLLBTranslator(device=self.device)
            # Ensure model and dynamic quantization are loaded into memory right now
            self._init_quantized_translator()
            LOGGER.info("Loaded NLLB-200 (Distilled 600M with int8 dynamic quantization).")
        except Exception as e:
            LOGGER.warning("Could not pre-load NLLBTranslator: %s", e)

        # Warmup execution
        self._warmup()
        t_elapsed = (time.perf_counter() - t_start) * 1000
        LOGGER.info("FastPipelineManager fully initialized in %.1f ms.", t_elapsed)

    def _init_quantized_translator(self) -> None:
        """Load NLLB-200 and apply int8 dynamic quantization on CPU if available."""
        if self.translator is None:
            return
        # If model is already loaded, skip
        if getattr(self.translator, "_model", None) is not None:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model_name = self.translator.model_name
            try:
                self.translator._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True).to(self.device)
            except Exception:
                # If local cache missing and offline, use fallback translator
                LOGGER.info("NLLB-200 not cached locally. Will use cached translation engines.")
                return

            # Apply int8 dynamic quantization to Linear layers on CPU
            if self.device.type == "cpu":
                try:
                    import torch.quantization
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    LOGGER.info("Applied int8 dynamic quantization to NLLB-200.")
                except Exception as q_err:
                    LOGGER.warning("Dynamic quantization not applied: %s", q_err)

            model.eval()
            self.translator._model = model
        except Exception as e:
            LOGGER.info("Offline or local environment: NLLB-200 will use cached translation fallback (%s)", e)

    def _warmup(self) -> None:
        """Run a minimal dummy inference pass to eliminate cold-start latency."""
        try:
            dummy = np.full((64, 128, 3), 255, dtype=np.uint8)
            if self.script_classifier:
                self.script_classifier.predict(dummy)
            if self.indic_crnn and self.indic_crnn.is_available:
                self.indic_crnn.recognize(dummy)
        except Exception as e:
            LOGGER.debug("Warmup pass non-critical warning: %s", e)

    def resize_for_speed(self, image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
        """Resize high-resolution image if either dimension exceeds max_dim, preserving aspect ratio."""
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / float(max(h, w))
        new_w = max(16, int(w * scale))
        new_h = max(16, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def batched_indic_crnn_recognize(
        self,
        line_crops: List[np.ndarray],
        max_batch_size: int = 16,
    ) -> List[Tuple[str, float]]:
        """Run batched forward pass on multiple line/word crops in a single tensor evaluation."""
        if not self.indic_crnn or not self.indic_crnn.is_available or not line_crops:
            return []

        results: List[Tuple[str, float]] = []

        # Process in batches
        for i in range(0, len(line_crops), max_batch_size):
            batch = line_crops[i : i + max_batch_size]
            tensors = []
            for crop in batch:
                t = self.indic_crnn.preprocess_image(crop)  # (1, 1, 32, max_w)
                tensors.append(t)

            batch_tensor = torch.cat(tensors, dim=0).to(self.device)  # (B, 1, 32, max_w)

            with torch.no_grad():
                log_probs = self.indic_crnn.model(batch_tensor)  # (T, B, num_classes)
                probs = torch.exp(log_probs)  # (T, B, num_classes)

                # Transpose to (B, T, num_classes)
                probs_b = probs.permute(1, 0, 2)
                for b_idx in range(probs_b.shape[0]):
                    seq_probs = probs_b[b_idx]  # (T, num_classes)
                    max_p, max_idx = torch.max(seq_probs, dim=-1)
                    indices = max_idx.cpu().tolist()
                    decoded = self.indic_crnn.charset.decode(indices, collapse_repeats=True, remove_blanks=True)

                    non_blank = max_idx != self.indic_crnn.charset.BLANK_INDEX
                    conf = float(max_p[non_blank].mean().item()) if non_blank.sum() > 0 else float(max_p.mean().item())
                    results.append((decoded.strip(), round(conf, 4)))

        return results

    def process(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        model_mode: str = "custom",  # 'custom' or 'baseline'
        script_hint: Optional[str] = None,
        input_type: str = "auto",  # 'auto', 'handwritten', 'printed'
        enable_cloud_bridge: bool = False,  # Strict offline translation by default
        confidence_threshold: float = 0.10,  # Confidence gate threshold before translation
        deskew: bool = True,
        denoise: bool = True,
        binarize: bool = True,
    ) -> FastPipelineResult:
        """Execute end-to-end inference with fine-grained per-stage wall-clock profiling."""
        timing = StageTiming()
        t_total_start = time.perf_counter()

        # -------------------------------------------------------------
        # STAGE 1: Image Ingestion & Preprocessing
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        raw_img = load_image(image_input)
        img = self.resize_for_speed(raw_img, max_dim=1280)

        ablation_cfg = AblationConfig(
            grayscale=True,
            denoise=denoise,
            deskew=deskew,
            binarize=binarize,
            binarize_method="sauvola",
            segment_lines=True,
            segment_words=False,
        )
        ablation_res = run_ablation_pipeline(img, config=ablation_cfg)
        processed_img = ablation_res.output_image
        extracted_lines = ablation_res.lines
        timing.preprocess_ms = (time.perf_counter() - t0) * 1000

        LOGGER.info(
            "[STAGE 1: PREPROCESS] input_shape=%s, processed_shape=%s, lines_detected=%d, timing=%.1fms",
            raw_img.shape,
            processed_img.shape,
            len(extracted_lines),
            timing.preprocess_ms,
        )

        # -------------------------------------------------------------
        # STAGE 2: Script Classification (PyTorch CNN)
        # -------------------------------------------------------------
        t1 = time.perf_counter()
        cnn_script = "Unknown"
        cnn_confidence = 0.50
        cnn_probs: Dict[str, float] = {}

        if self.script_classifier is not None:
            try:
                # Use a fast representative patch / downscaled probe
                h, w = processed_img.shape[:2]
                max_d = max(h, w)
                if max_d > 384:
                    scale = 384.0 / max_d
                    probe_img = cv2.resize(processed_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    probe_img = processed_img

                pred = self.script_classifier.predict(probe_img)
                cnn_script = pred.script
                cnn_confidence = float(pred.confidence)
                cnn_probs = pred.probabilities
            except Exception as e:
                LOGGER.warning("Visual script classification failed: %s", e)

        # Manual script hint override if specified by user
        effective_script = cnn_script
        if script_hint and script_hint.lower() not in ("auto", "none", "unknown"):
            for k in ("kannada", "tamil", "telugu", "arabic", "german", "english", "latin"):
                if k in script_hint.lower():
                    if k in ("german", "english"):
                        effective_script = "Latin"
                    else:
                        effective_script = k.capitalize()
                    break
        elif self.easyocr and (cnn_script == "Unknown" or cnn_confidence < 0.70):
            # Fast thumbnail probe across candidate scripts when CNN classifier is uncertain
            try:
                p_ar = self.easyocr.recognize(probe_img, lang="ar", canvas_size=384, batch_size=8)
                p_lat = self.easyocr.recognize(probe_img, lang="de", canvas_size=384, batch_size=8)
                p_kn = self.easyocr.recognize(probe_img, lang="kn", canvas_size=384, batch_size=8)
                p_te = self.easyocr.recognize(probe_img, lang="te", canvas_size=384, batch_size=8)
                p_ta = self.easyocr.recognize(probe_img, lang="ta", canvas_size=384, batch_size=8)

                ar_count = sum(1 for ch in (p_ar.text or "") if 0x0600 <= ord(ch) <= 0x06FF)
                lat_count = sum(1 for ch in (p_lat.text or "") if ('a' <= ch.lower() <= 'z') or ch in "äöüß")
                kn_count = sum(1 for ch in (p_kn.text or "") if 0x0C80 <= ord(ch) <= 0x0CFF)
                te_count = sum(1 for ch in (p_te.text or "") if 0x0C00 <= ord(ch) <= 0x0C7F)
                ta_count = sum(1 for ch in (p_ta.text or "") if 0x0B80 <= ord(ch) <= 0x0BFF)

                counts = {"Arabic": ar_count, "Kannada": kn_count, "Telugu": te_count, "Tamil": ta_count, "Latin": lat_count}
                best_probe_script, max_count = max(counts.items(), key=lambda x: x[1])

                if max_count >= 6:
                    effective_script = best_probe_script
                    LOGGER.info("Uncertain CNN resolved by probe: %s (%d chars)", effective_script, max_count)
            except Exception as probe_err:
                LOGGER.debug("Fast script probe non-critical error: %s", probe_err)

        timing.script_classifier_ms = (time.perf_counter() - t1) * 1000

        LOGGER.info(
            "[STAGE 2: SCRIPT CLASSIFIER] visual_script=%s, confidence=%.3f, effective_script=%s, timing=%.1fms",
            cnn_script,
            cnn_confidence,
            effective_script,
            timing.script_classifier_ms,
        )

        # -------------------------------------------------------------
        # STAGE 3: OCR Recognition (Batched CRNN vs. Baseline)
        # -------------------------------------------------------------
        t2 = time.perf_counter()
        recognized_text = ""
        ocr_confidence = 0.0
        engine_used = ""
        line_records: List[Dict[str, Any]] = []

        is_custom = model_mode.lower() in ("custom", "our_model", "our models")

        if is_custom and effective_script in ("Kannada", "Tamil", "Telugu") and self.indic_crnn and self.indic_crnn.is_available:
            engine_used = "Indic-CRNN (Shared Kannada-Tamil-Telugu)"
            # Extract line crops
            crops = []
            valid_boxes = []
            for idx, lr in enumerate(extracted_lines):
                x1, y1, x2, y2 = lr.bbox_xyxy
                crop = img[y1:y2, x1:x2]
                if crop.size > 0 and (x2 - x1) >= 8 and (y2 - y1) >= 8:
                    crops.append(crop)
                    valid_boxes.append((x1, y1, x2, y2))

            if crops:
                batch_outputs = self.batched_indic_crnn_recognize(crops)
                line_texts = []
                confs = []
                for idx, (txt, conf) in enumerate(batch_outputs):
                    if txt and txt.strip():
                        line_texts.append(txt.strip())
                        confs.append(conf)
                        line_records.append({
                            "line_num": idx + 1,
                            "text": txt.strip(),
                            "confidence": conf,
                            "bbox": valid_boxes[idx],
                        })
                recognized_text = "\n".join(line_texts).strip()
                ocr_confidence = float(np.mean(confs)) if confs else 0.0

            # Fallback to whole image CRNN if crops were empty or none recognized
            if not recognized_text:
                single_res = self.indic_crnn.recognize(img)
                recognized_text = single_res.text.strip()
                ocr_confidence = single_res.confidence

            # Fallback to EasyOCR if CRNN produced nothing
            if not recognized_text and self.easyocr:
                lang_code_map = {"Kannada": "kn", "Tamil": "ta", "Telugu": "te"}
                l_code = lang_code_map.get(effective_script, "kn")
                try:
                    easy_fb = self.easyocr.recognize(img, lang=l_code)
                    if easy_fb.text and easy_fb.text.strip():
                        recognized_text = easy_fb.text.strip()
                        ocr_confidence = easy_fb.confidence
                        engine_used = f"Fallback: EasyOCR ({effective_script})"
                except Exception as fb_err:
                    LOGGER.warning("EasyOCR fallback for %s failed: %s", effective_script, fb_err)

        elif is_custom and effective_script == "Latin" and self.trocr:
            chosen_key = "latin_printed" if input_type.lower() == "printed" else "latin_handwritten"
            engine_used = f"TrOCR ({'Latin Printed' if chosen_key == 'latin_printed' else 'Latin Handwritten'})"
            try:
                trocr_res = self.trocr.recognize(img, model_key=chosen_key)
                recognized_text = trocr_res.text.strip()
                ocr_confidence = trocr_res.confidence
                confs = trocr_res.raw_details.get("confidences", [])
                for idx, line_str in enumerate(recognized_text.split("\n")):
                    if line_str.strip():
                        c_val = float(confs[idx]) if idx < len(confs) else ocr_confidence
                        line_records.append({
                            "line_num": idx + 1,
                            "text": line_str.strip(),
                            "confidence": round(c_val, 3),
                            "is_low_confidence": c_val < 0.40,
                        })
            except Exception:
                recognized_text = ""

            # If TrOCR failed or returned too few characters on an illuminated/full page, fallback to EasyOCR German/Latin
            if len(recognized_text) < 15 and self.easyocr:
                try:
                    easy_res = self.easyocr.recognize(img, lang="de")
                    if len(easy_res.text.strip()) > len(recognized_text):
                        recognized_text = easy_res.text.strip()
                        ocr_confidence = easy_res.confidence
                        engine_used = "EasyOCR (German / Latin)"
                        line_records.clear()
                        boxes = easy_res.raw_details.get("boxes", [])
                        confs = easy_res.raw_details.get("confidences", [])
                        txt_lines = [l for l in recognized_text.split("\n") if l.strip()]
                        for idx, l_txt in enumerate(txt_lines):
                            c_val = float(confs[idx]) if idx < len(confs) else 0.5
                            box_val = None
                            if idx < len(boxes):
                                try:
                                    pts = np.array(boxes[idx], dtype=np.int32)
                                    box_val = (int(np.min(pts[:, 0])), int(np.min(pts[:, 1])), int(np.max(pts[:, 0])), int(np.max(pts[:, 1])))
                                except Exception:
                                    pass
                            line_records.append({
                                "line_num": idx + 1,
                                "text": l_txt,
                                "confidence": round(c_val, 3),
                                "is_low_confidence": c_val < 0.40,
                                "bbox": box_val,
                            })
                except Exception:
                    pass
        else:
            # Baseline Standard: EasyOCR (or Tesseract for printed German if requested)
            if input_type.lower() == "printed" and effective_script == "Latin" and getattr(self, "tesseract", None):
                tess_res = self.tesseract.recognize(img, lang="deu")
                if tess_res.text and len(tess_res.text.strip()) >= 10:
                    recognized_text = tess_res.text.strip()
                    ocr_confidence = tess_res.confidence
                    engine_used = "Tesseract Baseline (deu - Printed)"
                    for idx, l_txt in enumerate(recognized_text.split("\n")):
                        if l_txt.strip():
                            line_records.append({
                                "line_num": idx + 1,
                                "text": l_txt.strip(),
                                "confidence": round(ocr_confidence, 3),
                                "is_low_confidence": ocr_confidence < 0.40,
                            })

            if not recognized_text and self.easyocr:
                lang_code_map = {
                    "Kannada": "kn",
                    "Tamil": "ta",
                    "Telugu": "te",
                    "Arabic": "ar",
                    "Latin": "de",
                    "Devanagari": "hi",
                }
                lang_code = lang_code_map.get(effective_script, "en")
                engine_used = f"EasyOCR ({effective_script})"
                easy_res = self.easyocr.recognize(img, lang=lang_code)
                recognized_text = easy_res.text.strip()
                ocr_confidence = easy_res.confidence
                boxes = easy_res.raw_details.get("boxes", [])
                confs = easy_res.raw_details.get("confidences", [])
                txt_lines = [l for l in recognized_text.split("\n") if l.strip()]
                for idx, l_txt in enumerate(txt_lines):
                    c_val = float(confs[idx]) if idx < len(confs) else 0.5
                    box_val = None
                    if idx < len(boxes):
                        try:
                            pts = np.array(boxes[idx], dtype=np.int32)
                            box_val = (int(np.min(pts[:, 0])), int(np.min(pts[:, 1])), int(np.max(pts[:, 0])), int(np.max(pts[:, 1])))
                        except Exception:
                            pass
                    line_records.append({
                        "line_num": idx + 1,
                        "text": l_txt,
                        "confidence": round(c_val, 3),
                        "is_low_confidence": c_val < 0.40,
                        "bbox": box_val,
                    })

        # Auto-correction: if in auto mode and recognized text has low script coherence, check Arabic
        if (not script_hint or script_hint.lower() in ("auto", "none")) and effective_script != "Arabic" and self.easyocr:
            try:
                p_ar = self.easyocr.recognize(img, lang="ar", canvas_size=640)
                ar_count = sum(1 for ch in (p_ar.text or "") if 0x0600 <= ord(ch) <= 0x06FF)
                curr_chars = sum(1 for ch in recognized_text if ('a' <= ch.lower() <= 'z') or (0x0900 <= ord(ch) <= 0x097F) or (0x0C80 <= ord(ch) <= 0x0CFF) or (0x0C00 <= ord(ch) <= 0x0C7F) or (0x0B80 <= ord(ch) <= 0x0BFF))
                if ar_count >= 15 and ar_count > curr_chars * 1.5:
                    LOGGER.info("Auto-corrected script to Arabic (found %d Arabic chars vs %d %s chars)", ar_count, curr_chars, effective_script)
                    effective_script = "Arabic"
                    recognized_text = p_ar.text.strip()
                    ocr_confidence = p_ar.confidence
                    engine_used = "EasyOCR (Arabic - Auto Recovered)"
                    line_records.clear()
                    txt_lines = [l for l in recognized_text.split("\n") if l.strip()]
                    confs_ar = p_ar.raw_details.get("confidences", [])
                    for idx, l_txt in enumerate(txt_lines):
                        c_val = float(confs_ar[idx]) if idx < len(confs_ar) else 0.50
                        line_records.append({
                            "line_num": idx + 1,
                            "text": l_txt,
                            "confidence": round(c_val, 3),
                            "is_low_confidence": c_val < 0.10,
                        })
            except Exception as auto_ar_err:
                LOGGER.debug("Auto Arabic check: %s", auto_ar_err)

        timing.ocr_ms = (time.perf_counter() - t2) * 1000

        has_recognized_text = bool(recognized_text and recognized_text.strip())
        LOGGER.info(
            "[STAGE 3: OCR] engine=%s, recognized_text_len=%d, has_text=%s, timing=%.1fms, preview='%s'",
            engine_used,
            len(recognized_text),
            has_recognized_text,
            timing.ocr_ms,
            recognized_text[:60].replace('\n', ' ') if has_recognized_text else "(EMPTY)",
        )

        if not has_recognized_text:
            LOGGER.warning("[STAGE 3: OCR WARNING] Could not recognize text in this image.")

        # -------------------------------------------------------------
        # STAGE 4: Input Language Identification
        # -------------------------------------------------------------
        t3 = time.perf_counter()
        detected_lang = "Uncertain"
        detected_lang_code = "unk"
        lang_confidence = cnn_confidence

        if not has_recognized_text:
            detected_lang = "Uncertain"
            detected_lang_code = "unk"
            is_uncertain = True
            confidence_tier = "uncertain"
        else:
            if effective_script == "Kannada":
                detected_lang = "Kannada"
                detected_lang_code = "kn"
            elif effective_script == "Tamil":
                detected_lang = "Tamil"
                detected_lang_code = "ta"
            elif effective_script == "Telugu":
                detected_lang = "Telugu"
                detected_lang_code = "te"
            elif effective_script == "Arabic":
                detected_lang = "Arabic"
                detected_lang_code = "ar"
            elif effective_script == "Latin":
                de_or_en, latin_conf = distinguish_german_vs_english(recognized_text)
                if de_or_en == "de":
                    detected_lang = "German"
                    detected_lang_code = "de"
                else:
                    detected_lang = "English"
                    detected_lang_code = "en"
                lang_confidence = round(0.5 * cnn_confidence + 0.5 * latin_conf, 4)
            elif effective_script == "Devanagari":
                has_sanskrit = any(m in recognized_text for m in ["ः", "।", "॥", "ऽ", "अस्ति", "भवति", "सत्यम्", "धर्म", "ऋत", "वरुण", "मित्र", "देव", "ऋग्वेद"])
                if has_sanskrit:
                    detected_lang = "Sanskrit"
                    detected_lang_code = "sa"
                else:
                    detected_lang = "Hindi"
                    detected_lang_code = "hi"
            else:
                detected_lang = "Uncertain"
                detected_lang_code = "unk"

            is_uncertain = lang_confidence < 0.40
            confidence_tier = "uncertain" if is_uncertain else ("high" if lang_confidence >= 0.70 else "medium")

        timing.language_id_ms = (time.perf_counter() - t3) * 1000

        LOGGER.info(
            "[STAGE 4: LANGUAGE ID] detected_language=%s, code=%s, confidence=%.3f, is_uncertain=%s, timing=%.1fms",
            detected_lang,
            detected_lang_code,
            lang_confidence,
            is_uncertain,
            timing.language_id_ms,
        )

        # -------------------------------------------------------------
        # CONFIDENCE GATING BEFORE TRANSLATION
        # -------------------------------------------------------------
        from src.confidence_gate import gate_ocr_lines_for_translation, FALLBACK_UNRELIABLE_MESSAGE

        gate_res = gate_ocr_lines_for_translation(
            lines=line_records,
            raw_full_text=recognized_text,
            threshold=confidence_threshold,
            expected_script=effective_script,
        )
        line_records = gate_res.evaluated_lines

        # -------------------------------------------------------------
        # STAGE 5: Machine Translation (NLLB-200 Distilled Quantized)
        # -------------------------------------------------------------
        t4 = time.perf_counter()
        translated_text = ""
        meta = LANGUAGE_METADATA.get(detected_lang.lower(), {})
        native_name = meta.get("native_name", detected_lang)

        # Check if the whole image or all recognized lines are low confidence
        if gate_res.whole_image_low_confidence or not gate_res.can_translate:
            translated_text = FALLBACK_UNRELIABLE_MESSAGE
            LOGGER.warning(
                "[STAGE 5: TRANSLATION GATED] %s (accepted_lines=%d, total_lines=%d)",
                FALLBACK_UNRELIABLE_MESSAGE,
                gate_res.accepted_lines_count,
                gate_res.total_lines,
            )
        elif has_recognized_text and not is_uncertain and gate_res.text_for_translation:
            text_to_translate = gate_res.text_for_translation

            # Check epigraphical / historical manuscript resolution if matching known corpus folios
            classical_trans = None
            try:
                if detected_lang_code == "kn":
                    from src.classical_languages import resolve_classical_kannada_phrases
                    classical_trans = resolve_classical_kannada_phrases(text_to_translate)
                elif detected_lang_code == "de":
                    from src.classical_languages import resolve_early_modern_german
                    classical_trans = resolve_early_modern_german(text_to_translate)
                elif detected_lang_code in ("sa", "hi"):
                    from src.classical_languages import lookup_canonical_sanskrit
                    classical_trans = lookup_canonical_sanskrit(text_to_translate)
            except Exception as e:
                LOGGER.debug("Classical phrase resolver skipped: %s", e)

            if classical_trans:
                translated_text = classical_trans
            elif detected_lang_code == "en":
                translated_text = text_to_translate
            else:
                nllb_src = meta.get("nllb_code", detected_lang_code)
                if self.translator:
                    trans_res = self.translator.translate(
                        text=text_to_translate,
                        src_lang=nllb_src,
                        tgt_lang="eng_Latn",
                    )
                    translated_text = trans_res.get("translated_text", "")
                else:
                    try:
                        from src.translation import translate_text
                        c_res = translate_text(
                            text_to_translate,
                            src_lang=detected_lang_code,
                            tgt_lang="en",
                            force_offline=(not enable_cloud_bridge),
                            enable_cloud_bridge=enable_cloud_bridge,
                        )
                        translated_text = c_res.translated_text
                    except Exception:
                        translated_text = "[Translation engine offline]"

        timing.translation_ms = (time.perf_counter() - t4) * 1000
        timing.total_ms = (time.perf_counter() - t_total_start) * 1000

        LOGGER.info(
            "[STAGE 5: TRANSLATION] translated_text_len=%d, timing=%.1fms, total_time=%.1fms",
            len(translated_text),
            timing.translation_ms,
            timing.total_ms,
        )

        # User-facing status notice
        warning_msg = None
        if not has_recognized_text:
            warning_msg = "Could not recognize text in this image"
        elif gate_res.whole_image_low_confidence or not gate_res.can_translate:
            warning_msg = FALLBACK_UNRELIABLE_MESSAGE
        elif is_uncertain:
            warning_msg = "Low confidence: language identification flagged as uncertain."
        elif gate_res.notice:
            warning_msg = gate_res.notice

        return FastPipelineResult(
            success=True,
            detected_language=detected_lang if not is_uncertain else "Uncertain",
            detected_language_code=detected_lang_code if not is_uncertain else "unk",
            detected_script=effective_script,
            language_native_name=native_name,
            confidence=lang_confidence,
            is_uncertain=is_uncertain,
            confidence_tier=confidence_tier,
            recognized_text=recognized_text,
            translated_text=translated_text,
            engine_used=engine_used,
            timing=timing,
            lines=line_records,
            warning=warning_msg,
        )
