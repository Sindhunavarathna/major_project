"""5-class Manuscript Script Classification Module (PyTorch).

Classes:
- Kannada
- Tamil
- Telugu
- Arabic
- Latin
"""

from src.script_classifier.model import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ScriptClassifierCNN,
)
from src.script_classifier.predict import (
    ScriptClassifier,
    ScriptPrediction,
    predict_script,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "ScriptClassifierCNN",
    "ScriptClassifier",
    "ScriptPrediction",
    "predict_script",
]
