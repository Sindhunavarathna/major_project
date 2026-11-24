"""Unit tests for the new multilingual pipeline modules (language_id, translator, 5-language baseline)."""

import pytest
from src.language_id import distinguish_german_vs_english
from src.translator import NLLB_CODE_MAP, NLLBTranslator


def test_language_id_german_detection():
    text_de = "Es war ein schöner Sonntagmorgen in Dittersbach und wir tranken Kaffee."
    lang, conf = distinguish_german_vs_english(text_de)
    assert lang == "de"
    assert conf >= 0.70


def test_language_id_english_detection():
    text_en = "This is a handwritten note describing ancient history of the region."
    lang, conf = distinguish_german_vs_english(text_en)
    assert lang == "en"
    assert conf >= 0.60


def test_nllb_language_code_mappings():
    # Verify all 5 required languages map to official NLLB codes
    assert NLLB_CODE_MAP["kn"] == "kan_Knda"
    assert NLLB_CODE_MAP["ta"] == "tam_Taml"
    assert NLLB_CODE_MAP["te"] == "tel_Telu"
    assert NLLB_CODE_MAP["de"] == "deu_Latn"
    assert NLLB_CODE_MAP["ar"] == "arb_Arab"
    assert NLLB_CODE_MAP["en"] == "eng_Latn"


def test_nllb_identity_skip():
    t = NLLBTranslator()
    res = t.translate("Hello world", src_lang="eng_Latn", tgt_lang="eng_Latn")
    assert res["engine"] == "identity_skip"
    assert res["translated_text"] == "Hello world"
