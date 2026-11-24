"""Shared Indic CRNN OCR Engine for Kannada, Tamil, and Telugu handwriting.

Conforms to BaseOCREngine interface.
Performs greedy CTC decoding (collapsing repeats, then removing blanks).
"""

import os
from typing import Any, Optional
import cv2
import numpy as np
import torch

from src.indic_ocr.charset import UnifiedIndicCharset
from src.indic_ocr.model import IndicCRNN
from src.ocr.base import BaseOCREngine, OCRResult
from src.utils import get_logger, load_image

LOGGER = get_logger("indic_crnn_engine")

DEFAULT_MODEL_PATH = os.path.join("models", "indic_crnn", "best_indic_crnn.pth")
DEFAULT_VOCAB_PATH = os.path.join("models", "indic_crnn", "vocab.json")


class IndicCRNNEngine(BaseOCREngine):
    """Deep CRNN Engine for shared Kannada, Tamil, and Telugu handwriting transcription."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(name="Indic-CRNN (Shared Kannada-Tamil-Telugu)")
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path or DEFAULT_MODEL_PATH
        self.vocab_path = vocab_path or DEFAULT_VOCAB_PATH

        self.charset: Optional[UnifiedIndicCharset] = None
        self.model: Optional[IndicCRNN] = None
        self._load_engine()

    def _load_engine(self) -> None:
        """Load vocabulary and trained model weights if present on disk."""
        if os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                if "vocab" in ckpt:
                    self.charset = UnifiedIndicCharset(custom_chars=ckpt["vocab"])
                elif os.path.exists(self.vocab_path):
                    self.charset = UnifiedIndicCharset.load(self.vocab_path)
                else:
                    self.charset = UnifiedIndicCharset()

                model = IndicCRNN(num_classes=len(self.charset))
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(self.device)
                model.eval()
                self.model = model
                LOGGER.info("Successfully loaded Indic-CRNN model from %s", self.checkpoint_path)
            except Exception as e:
                LOGGER.warning("Failed to initialize Indic-CRNN from checkpoint: %s", e)
                self.model = None
                self.charset = None
        else:
            LOGGER.info("No Indic-CRNN checkpoint at %s; engine uninitialized.", self.checkpoint_path)

    @property
    def is_available(self) -> bool:
        """Check whether the model weights and charset are loaded and ready."""
        return self.model is not None and self.charset is not None

    def preprocess_image(self, image: Any, target_height: int = 32, max_width: int = 256) -> torch.Tensor:
        """Convert image or file path to normalized 1x1x32xW float tensor."""
        img = load_image(image)
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape[:2]
        if h != target_height:
            new_w = max(16, int(w * (target_height / float(h))))
            gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_AREA)

        h, w = gray.shape[:2]
        if w > max_width:
            gray = cv2.resize(gray, (max_width, target_height), interpolation=cv2.INTER_AREA)
            w = max_width

        canvas = np.full((target_height, max_width), 255, dtype=np.uint8)
        canvas[:, :w] = gray

        norm = (canvas.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # (1, 1, 32, max_width)
        return tensor.to(self.device)

    def recognize(
        self,
        image: np.ndarray,
        lang: Optional[str] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Perform handwriting recognition on the input image using the shared Indic CRNN."""
        if not self.is_available:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                is_fallback=True,
                warning="Indic-CRNN model weights not loaded. Run training script first.",
                raw_details={"status": "checkpoint_missing"},
            )

        tensor = self.preprocess_image(image)
        with torch.no_grad():
            log_probs = self.model(tensor)  # (T, 1, num_classes)
            probs = torch.exp(log_probs).squeeze(1)  # (T, num_classes)
            max_probs, max_indices = torch.max(probs, dim=1)

            # Greedy CTC decoding: collapse consecutive repeats, then drop blanks
            indices_list = max_indices.cpu().tolist()
            decoded_text = self.charset.decode(indices_list, collapse_repeats=True, remove_blanks=True)

            non_blank_mask = max_indices != self.charset.BLANK_INDEX
            if non_blank_mask.sum() > 0:
                conf = float(max_probs[non_blank_mask].mean().item())
            else:
                conf = float(max_probs.mean().item())

        return OCRResult(
            text=decoded_text.strip(),
            confidence=round(conf, 4),
            engine=self.name,
            is_fallback=False,
            raw_details={
                "language_hint": lang,
                "token_sequence_length": len(indices_list),
            },
        )
