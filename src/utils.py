"""Utility functions for logging, device detection, image I/O, and data formatting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image
import torch


def get_logger(name: str = "manuscript_ocr") -> logging.Logger:
    """Return a configured logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = get_logger("core_utils")


def get_torch_device() -> torch.device:
    """Detect available accelerator device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name() -> str:
    """Return string representation of active device."""
    dev = get_torch_device()
    return str(dev.type)


def load_image(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image]
) -> np.ndarray:
    """Load an image from various input types and return a standard BGR numpy array.

    Args:
        image_input: File path, byte stream, PIL Image, or existing numpy array.

    Returns:
        np.ndarray: BGR image array suitable for OpenCV operations.
    """
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            import cv2
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        if image_input.shape[2] == 4:
            import cv2
            return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        return image_input.copy()

    if isinstance(image_input, Image.Image):
        rgb_img = image_input.convert("RGB")
        import cv2
        return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

    if isinstance(image_input, bytes):
        import cv2
        arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from provided byte stream.")
        return img

    if isinstance(image_input, (str, Path)):
        import cv2
        path_str = str(image_input)
        img = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at path: {path_str}")
        return img

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image array to PIL RGB Image."""
    import cv2
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def format_json_result(
    original_text: str,
    detected_lang: str,
    detected_script: str,
    lang_confidence: float,
    translation: str,
    target_lang: str,
    engine_used: str,
    ocr_confidence: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Format OCR and translation pipeline output into a clean JSON string."""
    payload = {
        "status": "success",
        "ocr": {
            "engine": engine_used,
            "extracted_text": original_text,
            "confidence": round(ocr_confidence, 4),
        },
        "language_identification": {
            "detected_language": detected_lang,
            "detected_script": detected_script,
            "confidence": round(lang_confidence, 4),
        },
        "translation": {
            "target_language": target_lang,
            "translated_text": translation,
        },
        "metadata": metadata or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

