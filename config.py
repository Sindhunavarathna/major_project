"""Configuration module for Multilingual Manuscript OCR, LangID, and Translation.

Loads settings from config.yaml or provides robust defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class TrOCRConfig:
    sanskrit_model_path: str = "models/sanskrit_trocr"
    latin_handwritten: str = "microsoft/trocr-base-handwritten"
    latin_printed: str = "microsoft/trocr-base-printed"
    base_model_name: str = "microsoft/trocr-base-handwritten"
    fallback_model_name: str = "microsoft/trocr-small-handwritten"


@dataclass
class TranslationConfig:
    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"
    max_length: int = 256
    num_beams_cpu: int = 1
    num_beams_gpu: int = 4


@dataclass
class LanguageClassifierConfig:
    model_path: str = "models/language_classifier/language_nb.pkl"


@dataclass
class DebugConfig:
    enabled: bool = False
    output_dir: str = "debug"


@dataclass
class DetectionConfig:
    min_crop_height: int = 12
    min_crop_width: int = 25
    min_crop_area: int = 300
    min_detection_confidence: float = 0.25
    padding_px: int = 6
    line_merge_vertical_overlap_ratio: float = 0.40


@dataclass
class QualityGatingConfig:
    min_characters: int = 4
    min_letter_ratio: float = 0.45
    min_mean_token_confidence: float = 0.35
    # EasyOCR word-level confidence is calibrated differently from TrOCR token probabilities.
    # Raw EasyOCR scores on multi-script content are typically 0.20–0.50, so we use a
    # separate, lower threshold rather than the TrOCR beam-softmax threshold.
    easyocr_min_confidence: float = 0.25


@dataclass
class EasyOCRConfig:
    gpu: str = "auto"
    languages: List[str] = field(default_factory=lambda: ["en", "hi", "kn"])


@dataclass
class TesseractConfig:
    tesseract_cmd: str = "tesseract"
    default_lang: str = "eng"
    oem: int = 3
    psm: int = 3


@dataclass
class OCRConfig:
    confidence_threshold: float = 0.35
    low_confidence_threshold: float = 0.50
    easyocr: EasyOCRConfig = field(default_factory=EasyOCRConfig)
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)


@dataclass
class PreprocessingConfig:
    denoise_strength: int = 10
    adaptive_block_size: int = 15
    adaptive_c: int = 4
    max_deskew_angle: float = 45.0
    min_line_height: int = 16


@dataclass
class LangIDConfig:
    min_text_len: int = 2
    confidence_threshold: float = 0.50
    script_detection: bool = True


@dataclass
class LanguagesConfig:
    supported_ocr: Dict[str, str] = field(
        default_factory=lambda: {
            "en": "English",
            "hi": "Hindi",
            "kn": "Kannada",
            "sa": "Sanskrit",
            "ta": "Tamil",
            "te": "Telugu",
            "ar": "Arabic",
            "mr": "Marathi",
            "bn": "Bengali",
        }
    )
    mbart_codes: Dict[str, str] = field(
        default_factory=lambda: {
            "en": "en_XX",
            "hi": "hi_IN",
            "ta": "ta_IN",
            "te": "te_IN",
            "mr": "mr_IN",
            "bn": "bn_IN",
            "gu": "gu_IN",
            "ml": "ml_IN",
            "ur": "ur_PK",
            "ne": "ne_NP",
            "si": "si_LK",
            "fr": "fr_XX",
            "de": "de_DE",
            "es": "es_XX",
            "ru": "ru_RU",
            "ar": "ar_AR",
            "zh": "zh_CN",
            "ja": "ja_XX",
        }
    )
    unsupported_mbart: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "sa": {
                "name": "Sanskrit",
                "default_pivot": "hi",
                "note": "Sanskrit is not in standard mBART-50. Hindi pivot or fine-tuning is required.",
            },
            "kn": {
                "name": "Kannada",
                "default_pivot": None,
                "note": "Kannada is not in standard mBART-50. Fine-tuning via scripts/finetune_mbart.py is required.",
            },
        }
    )


@dataclass
class AppConfig:
    trocr: TrOCRConfig = field(default_factory=TrOCRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    language_classifier: LanguageClassifierConfig = field(default_factory=LanguageClassifierConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    quality_gating: QualityGatingConfig = field(default_factory=QualityGatingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    langid: LangIDConfig = field(default_factory=LangIDConfig)
    languages: LanguagesConfig = field(default_factory=LanguagesConfig)

    def is_mbart_supported(self, lang_code: str) -> bool:
        """Check if language code is directly supported by mBART-50."""
        return lang_code.lower() in self.languages.mbart_codes

    def get_mbart_code(self, lang_code: str) -> Optional[str]:
        """Get 5-character mBART code (e.g. 'en' -> 'en_XX')."""
        return self.languages.mbart_codes.get(lang_code.lower())

    def get_unsupported_info(self, lang_code: str) -> Optional[Dict[str, Any]]:
        """Retrieve notes and fallback strategies for unsupported languages."""
        return self.languages.unsupported_mbart.get(lang_code.lower())


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from a YAML file with dataclass validation and defaults."""
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    data: Dict[str, Any] = {}

    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}

    models_data = data.get("models", {})
    trocr_data = models_data.get("trocr", {})
    trans_data = models_data.get("translation", {})
    lang_clf_data = models_data.get("language_classifier", {})

    debug_data = data.get("debug", {})
    det_data = data.get("detection", {})
    qg_data = data.get("quality_gating", {})

    ocr_data = data.get("ocr", {})
    easyocr_data = ocr_data.get("easyocr", {})
    tess_data = ocr_data.get("tesseract", {})

    # Detect tesseract path if default is not valid
    tess_cmd = tess_data.get("tesseract_cmd", "tesseract")
    if not os.path.exists(tess_cmd):
        alt_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                tess_cmd = p
                break

    prep_data = data.get("preprocessing", {})
    langid_data = data.get("langid", {})
    langs_data = data.get("languages", {})

    return AppConfig(
        trocr=TrOCRConfig(
            sanskrit_model_path=trocr_data.get("sanskrit_model_path", "models/sanskrit_trocr"),
            latin_handwritten=trocr_data.get("latin_handwritten", "microsoft/trocr-base-handwritten"),
            latin_printed=trocr_data.get("latin_printed", "microsoft/trocr-base-printed"),
            base_model_name=trocr_data.get("base_model_name", "microsoft/trocr-base-handwritten"),
            fallback_model_name=trocr_data.get("fallback_model_name", "microsoft/trocr-small-handwritten"),
        ),
        translation=TranslationConfig(
            model_name=trans_data.get("model_name", "facebook/mbart-large-50-many-to-many-mmt"),
            max_length=int(trans_data.get("max_length", 256)),
            num_beams_cpu=int(trans_data.get("num_beams_cpu", 1)),
            num_beams_gpu=int(trans_data.get("num_beams_gpu", 4)),
        ),
        language_classifier=LanguageClassifierConfig(
            model_path=lang_clf_data.get("model_path", "models/language_classifier/language_nb.pkl"),
        ),
        debug=DebugConfig(
            enabled=bool(debug_data.get("enabled", False)),
            output_dir=str(debug_data.get("output_dir", "debug")),
        ),
        detection=DetectionConfig(
            min_crop_height=int(det_data.get("min_crop_height", 12)),
            min_crop_width=int(det_data.get("min_crop_width", 25)),
            min_crop_area=int(det_data.get("min_crop_area", 300)),
            min_detection_confidence=float(det_data.get("min_detection_confidence", 0.25)),
            padding_px=int(det_data.get("padding_px", 6)),
            line_merge_vertical_overlap_ratio=float(det_data.get("line_merge_vertical_overlap_ratio", 0.40)),
        ),
        quality_gating=QualityGatingConfig(
            min_characters=int(qg_data.get("min_characters", 4)),
            min_letter_ratio=float(qg_data.get("min_letter_ratio", 0.45)),
            min_mean_token_confidence=float(qg_data.get("min_mean_token_confidence", 0.60)),
            easyocr_min_confidence=float(qg_data.get("easyocr_min_confidence", 0.25)),
        ),
        ocr=OCRConfig(
            confidence_threshold=float(ocr_data.get("confidence_threshold", 0.35)),
            low_confidence_threshold=float(ocr_data.get("low_confidence_threshold", 0.50)),
            easyocr=EasyOCRConfig(
                gpu=str(easyocr_data.get("gpu", "auto")),
                languages=list(easyocr_data.get("languages", ["en", "hi", "kn"])),
            ),
            tesseract=TesseractConfig(
                tesseract_cmd=tess_cmd,
                default_lang=str(tess_data.get("default_lang", "eng")),
                oem=int(tess_data.get("oem", 3)),
                psm=int(tess_data.get("psm", 3)),
            ),
        ),
        preprocessing=PreprocessingConfig(
            denoise_strength=int(prep_data.get("denoise_strength", 10)),
            adaptive_block_size=int(prep_data.get("adaptive_block_size", 15)),
            adaptive_c=int(prep_data.get("adaptive_c", 4)),
            max_deskew_angle=float(prep_data.get("max_deskew_angle", 45.0)),
            min_line_height=int(prep_data.get("min_line_height", 16)),
        ),
        langid=LangIDConfig(
            min_text_len=int(langid_data.get("min_text_len", 2)),
            confidence_threshold=float(langid_data.get("confidence_threshold", 0.50)),
            script_detection=bool(langid_data.get("script_detection", True)),
        ),
        languages=LanguagesConfig(
            supported_ocr=langs_data.get(
                "supported_ocr",
                {
                    "en": "English",
                    "hi": "Hindi",
                    "kn": "Kannada",
                    "sa": "Sanskrit",
                    "ta": "Tamil",
                    "te": "Telugu",
                    "ar": "Arabic",
                    "mr": "Marathi",
                    "bn": "Bengali",
                },
            ),
            mbart_codes=langs_data.get(
                "mbart_codes",
                {
                    "en": "en_XX",
                    "hi": "hi_IN",
                    "ta": "ta_IN",
                    "te": "te_IN",
                    "mr": "mr_IN",
                    "bn": "bn_IN",
                    "gu": "gu_IN",
                    "ml": "ml_IN",
                    "ur": "ur_PK",
                    "ne": "ne_NP",
                    "si": "si_LK",
                    "fr": "fr_XX",
                    "de": "de_DE",
                    "es": "es_XX",
                    "ru": "ru_RU",
                    "ar": "ar_AR",
                    "zh": "zh_CN",
                    "ja": "ja_XX",
                },
            ),
            unsupported_mbart=langs_data.get(
                "unsupported_mbart",
                {
                    "sa": {
                        "name": "Sanskrit",
                        "default_pivot": "hi",
                        "note": "Sanskrit is not in standard mBART-50. Hindi pivot or fine-tuning is required.",
                    },
                    "kn": {
                        "name": "Kannada",
                        "default_pivot": None,
                        "note": "Kannada is not in standard mBART-50. Fine-tuning via scripts/finetune_mbart.py is required.",
                    },
                },
            ),
        ),
    )


CONFIG: AppConfig = load_config()
