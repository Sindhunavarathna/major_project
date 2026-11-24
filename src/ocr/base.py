"""Base classes and data structures for OCR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class OCRResult:
    """Standardized representation of OCR output."""
    text: str
    confidence: float
    engine: str
    is_fallback: bool = False
    warning: Optional[str] = None
    raw_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine,
            "is_fallback": self.is_fallback,
            "warning": self.warning,
            "raw_details": self.raw_details,
        }


class BaseOCREngine(ABC):
    """Abstract base class for all OCR engine wrappers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        lang: Optional[str] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Run OCR recognition on the input image.

        Args:
            image: Standard BGR or Grayscale numpy array.
            lang: Optional language code hint (e.g. 'en', 'hi', 'sa', 'kn').
            **kwargs: Engine-specific parameters.

        Returns:
            OCRResult containing extracted text, confidence, and metadata.
        """
        pass

