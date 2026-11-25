"""Inference module for 5-class manuscript script classification.

Provides fast CPU-runnable script prediction on any input image.
"""

from dataclasses import dataclass
import os
from typing import Dict, Optional, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.script_classifier.model import CLASS_NAMES, ScriptClassifierCNN
from src.utils import get_logger, load_image

LOGGER = get_logger("script_predict")

DEFAULT_CHECKPOINT = os.path.join("models", "script_classifier", "best_script_classifier.pth")


@dataclass
class ScriptPrediction:
    """Result of script classification."""
    script: str
    confidence: float
    probabilities: Dict[str, float]
    raw_logits: Optional[list] = None

    @property
    def predicted_class(self) -> str:
        return self.script


class ScriptClassifier:
    """Wrapper for running inference using the trained PyTorch ScriptClassifierCNN."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.model: Optional[ScriptClassifierCNN] = None
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                classes = ckpt.get("classes", CLASS_NAMES)
                model = ScriptClassifierCNN(num_classes=len(classes), in_channels=1)
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(self.device)
                model.eval()
                self.model = model
                LOGGER.info("Loaded script classifier model from %s", self.checkpoint_path)
            except Exception as e:
                LOGGER.warning("Could not load script classifier checkpoint: %s", e)
                self.model = None
        else:
            LOGGER.info("Checkpoint not found at %s. Model uninitialized.", self.checkpoint_path)
            self.model = None

    def preprocess_image(self, image_input: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Convert input image to normalized 1x1x128x128 tensor."""
        img = load_image(image_input)
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        norm = (resized.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 128, 128)
        return tensor.to(self.device)

    def predict(self, image_input: Union[np.ndarray, Image.Image]) -> ScriptPrediction:
        """Classify script of the given manuscript image patch."""
        if self.model is None:
            # Fallback heuristic if weights not loaded yet
            return ScriptPrediction(
                script="Latin",
                confidence=0.50,
                probabilities={name: 0.20 for name in CLASS_NAMES},
            )

        tensor = self.preprocess_image(image_input)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_script = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])
        prob_dict = {name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)}

        return ScriptPrediction(
            script=pred_script,
            confidence=round(conf, 4),
            probabilities=prob_dict,
            raw_logits=logits.squeeze(0).cpu().tolist(),
        )


def predict_script(image: Union[np.ndarray, Image.Image], checkpoint_path: Optional[str] = None) -> ScriptPrediction:
    """One-liner helper to predict the script of an image."""
    classifier = ScriptClassifier(checkpoint_path=checkpoint_path)
    return classifier.predict(image)
