"""NLLB-200 Translation Module for Multilingual Handwritten Note Recognition.

Supports the 5 required project languages:
- Kannada: kan_Knda
- Tamil: tam_Taml
- Telugu: tel_Telu
- German: deu_Latn
- Arabic: arb_Arab
Target language:
- English: eng_Latn
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
import torch

from src.utils import get_logger, get_torch_device

LOGGER = get_logger("translator")

# Official NLLB-200 BCP-47 language codes
NLLB_CODE_MAP = {
    "kn": "kan_Knda",
    "kannada": "kan_Knda",
    "kan_knda": "kan_Knda",
    
    "ta": "tam_Taml",
    "tamil": "tam_Taml",
    "tam_taml": "tam_Taml",
    
    "te": "tel_Telu",
    "telugu": "tel_Telu",
    "tel_telu": "tel_Telu",
    
    "de": "deu_Latn",
    "german": "deu_Latn",
    "deutsch": "deu_Latn",
    "deu_latn": "deu_Latn",
    
    "ar": "arb_Arab",
    "arabic": "arb_Arab",
    "arb_arab": "arb_Arab",
    
    "en": "eng_Latn",
    "english": "eng_Latn",
    "eng_latn": "eng_Latn",
}

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"


class NLLBTranslator:
    """Wrapper for Facebook NLLB-200 distilled translation with lazy initialization."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[torch.device] = None):
        self.model_name = model_name
        self.device = device or get_torch_device()
        self._tokenizer = None
        self._model = None

    def _ensure_model_loaded(self) -> bool:
        """Lazy loader for NLLB-200 model weights and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return True

        LOGGER.info("Checking NLLB-200 (%s) on %s...", self.model_name, self.device)
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True).to(self.device)
            except Exception:
                LOGGER.info("NLLB-200 not found in local cache. Using local mBART/classical fallback.")
                return False

            if self.device.type == "cpu":
                try:
                    import torch.quantization
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    LOGGER.info("Applied int8 dynamic quantization to NLLB-200 on CPU.")
                except Exception as q_err:
                    LOGGER.warning("NLLB dynamic quantization skipped: %s", q_err)
            self._model = model
            self._model.eval()
            LOGGER.info("NLLB-200 loaded successfully.")
            return True
        except Exception as e:
            LOGGER.warning("Could not load NLLB-200 (%s): %s. Will use fallback.", self.model_name, e)
            return False

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str = "eng_Latn",
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """Translate text using NLLB-200 with fallback to existing translation engines.

        Args:
            text: Source text to translate.
            src_lang: Source language code (e.g., 'kn', 'ta', 'te', 'de', 'ar' or 'kan_Knda').
            tgt_lang: Target language code (default: 'eng_Latn').
            max_length: Maximum generated token sequence length.

        Returns:
            Dict containing translated_text, src_lang, tgt_lang, and engine info.
        """
        if not text or not text.strip():
            return {
                "translated_text": "",
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "engine": "none",
            }

        # Resolve standard ISO / shorthand to official NLLB codes
        clean_src = NLLB_CODE_MAP.get(src_lang.lower().strip(), "kan_Knda")
        clean_tgt = NLLB_CODE_MAP.get(tgt_lang.lower().strip(), "eng_Latn")

        # Identity skip: same language
        if clean_src == clean_tgt:
            return {
                "translated_text": text.strip(),
                "src_lang": clean_src,
                "tgt_lang": clean_tgt,
                "engine": "identity_skip",
            }

        # Try NLLB-200 first
        if self._ensure_model_loaded():
            try:
                # Set source language on tokenizer
                self._tokenizer.src_lang = clean_src
                inputs = self._tokenizer(text.strip(), return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(clean_tgt)
                with torch.no_grad():
                    generated_tokens = self._model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_new_tokens=max_length,
                        num_beams=1,  # Greedy for low CPU latency
                        early_stopping=True,
                    )
                out_text = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                return {
                    "translated_text": out_text.strip(),
                    "src_lang": clean_src,
                    "tgt_lang": clean_tgt,
                    "engine": "NLLB-200 (facebook/nllb-200-distilled-600M)",
                }
            except Exception as e:
                LOGGER.warning("NLLB-200 generation error: %s. Falling back to primary engine.", e)

        # Fallback to existing project translation engine (mBART-50 / Google GTX / Canonical)
        from src.translation import translate_text as legacy_translate
        iso_map = {"kan_Knda": "kn", "tam_Taml": "ta", "tel_Telu": "te", "deu_Latn": "de", "arb_Arab": "ar", "eng_Latn": "en"}
        short_src = iso_map.get(clean_src, "auto")
        short_tgt = iso_map.get(clean_tgt, "en")
        
        fallback_res = legacy_translate(
            text=text,
            src_lang=short_src,
            tgt_lang=short_tgt,
            force_offline=True,
            enable_cloud_bridge=False,
        )
        return {
            "translated_text": fallback_res.translated_text,
            "src_lang": clean_src,
            "tgt_lang": clean_tgt,
            "engine": f"Fallback: {fallback_res.model_name}",
        }


# Global singleton instance
_GLOBAL_NLLB_TRANSLATOR: Optional[NLLBTranslator] = None

def get_nllb_translator() -> NLLBTranslator:
    global _GLOBAL_NLLB_TRANSLATOR
    if _GLOBAL_NLLB_TRANSLATOR is None:
        _GLOBAL_NLLB_TRANSLATOR = NLLBTranslator()
    return _GLOBAL_NLLB_TRANSLATOR
