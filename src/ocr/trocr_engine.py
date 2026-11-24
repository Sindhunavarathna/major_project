"""TrOCR engine for manuscript and handwritten text recognition.

Supports script-aware models:
- Latin handwritten: microsoft/trocr-base-handwritten
- Latin printed: microsoft/trocr-base-printed
- Sanskrit/Devanagari: local fine-tuned TrOCR at models/sanskrit_trocr/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from config import CONFIG
from src.ocr.base import BaseOCREngine, OCRResult
from src.preprocessing import segment_lines
from src.utils import bgr_to_pil, get_logger, get_torch_device

LOGGER = get_logger("trocr_engine")


class TrOCREngine(BaseOCREngine):
    """Script-aware TrOCR wrapper supporting multiple models and explicit fallback management."""

    def __init__(
        self,
        sanskrit_model_path: Optional[str] = None,
        base_model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(name="TrOCR")
        self.sanskrit_model_path = Path(
            sanskrit_model_path or CONFIG.trocr.sanskrit_model_path
        )
        self.base_model_name = base_model_name or CONFIG.trocr.base_model_name
        self.device = device or get_torch_device()

        # Cache models per model identifier
        self._processors: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        self.is_fallback: bool = False
        self.warning: Optional[str] = None

    def _is_valid_local_model(self, path: Path) -> bool:
        """Check if local directory exists and contains valid model weights."""
        if not path.exists() or not path.is_dir():
            return False
        has_config = (path / "config.json").exists()
        has_weights = (
            any(path.glob("*.safetensors"))
            or any(path.glob("*.bin"))
            or (path / "pytorch_model.bin").exists()
        )
        return has_config and has_weights

    def get_model_and_processor(
        self, model_key: str = "latin_handwritten"
    ) -> Tuple[Any, Any, str, bool, Optional[str]]:
        """Retrieve model and processor based on model_key or script request."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        clean_key = model_key.lower().strip()
        is_fallback = False
        warning = None

        if clean_key in ("sanskrit", "devanagari"):
            if self._is_valid_local_model(self.sanskrit_model_path):
                target_path = str(self.sanskrit_model_path)
            else:
                target_path = None
                is_fallback = True
                warning = (
                    f"Fine-tuned Sanskrit TrOCR model not found at '{self.sanskrit_model_path}'. "
                    f"Devanagari handwriting recognition requires these weights. "
                    f"Place the checkpoint in '{self.sanskrit_model_path}'."
                )
        elif clean_key in ("latin_printed", "english_printed"):
            target_path = CONFIG.trocr.latin_printed
        else:
            # Default to Latin handwritten
            target_path = CONFIG.trocr.latin_handwritten

        if target_path is None:
            return None, None, clean_key, is_fallback, warning

        if target_path not in self._models:
            LOGGER.info("Loading TrOCR model '%s' on %s...", target_path, self.device)
            try:
                try:
                    proc = TrOCRProcessor.from_pretrained(target_path, local_files_only=True)
                    mod = VisionEncoderDecoderModel.from_pretrained(target_path, local_files_only=True)
                except Exception:
                    proc = TrOCRProcessor.from_pretrained(target_path)
                    mod = VisionEncoderDecoderModel.from_pretrained(target_path)
                mod.to(self.device)
                mod.eval()
                self._processors[target_path] = proc
                self._models[target_path] = mod
            except Exception as exc:
                LOGGER.error("Failed to load TrOCR from '%s': %s", target_path, exc)
                # Fallback to base model
                fallback_target = CONFIG.trocr.base_model_name
                if target_path != fallback_target:
                    LOGGER.info("Falling back to '%s'...", fallback_target)
                    try:
                        proc = TrOCRProcessor.from_pretrained(fallback_target, local_files_only=True)
                        mod = VisionEncoderDecoderModel.from_pretrained(fallback_target, local_files_only=True)
                    except Exception:
                        proc = TrOCRProcessor.from_pretrained(fallback_target)
                        mod = VisionEncoderDecoderModel.from_pretrained(fallback_target)
                    mod.to(self.device)
                    mod.eval()
                    self._processors[fallback_target] = proc
                    self._models[fallback_target] = mod
                    target_path = fallback_target
                    is_fallback = True
                    warning = f"Fell back to base TrOCR model '{fallback_target}': {exc}"
                else:
                    raise

        return self._models[target_path], self._processors[target_path], target_path, is_fallback, warning

    def _infer_single_image(
        self, pil_img: Image.Image, model: Any, processor: Any
    ) -> Tuple[str, float]:
        """Perform TrOCR line recognition and calculate mean token confidence."""
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_length=64,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        confidence = 0.80
        if outputs.scores and len(outputs.scores) > 0:
            probs = []
            for idx, score_step in enumerate(outputs.scores):
                step_softmax = torch.softmax(score_step, dim=-1)
                token_id = (
                    generated_ids[0, idx + 1]
                    if (idx + 1) < generated_ids.shape[1]
                    else None
                )
                if token_id is not None and token_id < step_softmax.shape[-1]:
                    prob = float(step_softmax[0, token_id].item())
                    probs.append(prob)
            if probs:
                confidence = float(np.mean(probs))

        return text, round(confidence, 4)

    def recognize_lines(
        self,
        line_crops: List[np.ndarray],
        model_key: str = "latin_handwritten",
    ) -> OCRResult:
        """Recognize an explicit list of pre-cropped horizontal text lines."""
        model, processor, active_path, is_fallback, warning = self.get_model_and_processor(model_key)

        if model is None or processor is None:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                is_fallback=is_fallback,
                warning=warning or "TrOCR model unavailable.",
                raw_details={"error": "model_unavailable", "model_key": model_key},
            )

        recognized_lines = []
        confidences = []

        for crop in line_crops:
            if crop.size == 0:
                continue
            pil_crop = bgr_to_pil(crop)
            line_text, conf = self._infer_single_image(pil_crop, model, processor)
            if line_text:
                recognized_lines.append(line_text)
                confidences.append(conf)

        full_text = "\n".join(recognized_lines)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=round(avg_conf, 4),
            engine=self.name,
            is_fallback=is_fallback,
            warning=warning,
            raw_details={
                "model_used": active_path,
                "model_key": model_key,
                "line_count": len(recognized_lines),
                "confidences": confidences,
            },
        )

    def recognize(
        self,
        image: np.ndarray,
        lang: Optional[str] = None,
        model_key: str = "latin_handwritten",
        segment_first: bool = True,
        **kwargs: Any,
    ) -> OCRResult:
        """Run TrOCR recognition on the image, segmenting into lines if multi-line."""
        # Determine model_key from lang hint if provided
        if lang in ("sa", "hi", "devanagari"):
            chosen_key = "sanskrit"
        elif lang in ("en_printed", "printed"):
            chosen_key = "latin_printed"
        else:
            chosen_key = model_key

        if segment_first:
            crops = segment_lines(image, min_line_height=CONFIG.preprocessing.min_line_height)
        else:
            crops = [image]

        return self.recognize_lines(crops, model_key=chosen_key)
