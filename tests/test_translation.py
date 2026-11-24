"""Unit tests for translation language code mapping, unsupported language handling, and chunking."""

from unittest.mock import MagicMock, patch

from src.translation import MultilingualTranslator


def test_language_code_resolution():
    trans = MultilingualTranslator()

    # Supported direct mapping
    code_en, pivot_en, warn_en = trans._resolve_lang_code("en")
    assert code_en == "en_XX"
    assert pivot_en is False
    assert warn_en is None

    code_hi, pivot_hi, warn_hi = trans._resolve_lang_code("hi")
    assert code_hi == "hi_IN"
    assert pivot_hi is False

    # Sanskrit pivot handling
    code_sa, pivot_sa, warn_sa = trans._resolve_lang_code("sa")
    assert code_sa == "hi_IN"
    assert pivot_sa is True
    assert "Hindi pivot" in warn_sa

    # Kannada unsupported handling
    code_kn, pivot_kn, warn_kn = trans._resolve_lang_code("kn")
    assert code_kn is None
    assert pivot_kn is False
    assert "Kannada is not pre-trained in standard mBART-50" in warn_kn


def test_translation_same_language_skipped():
    trans = MultilingualTranslator()
    res = trans.translate("Hello world", src_lang="en", tgt_lang="en")
    assert res.is_skipped is True
    assert res.translated_text == "Hello world"
    assert "identical" in res.warning.lower()


def test_translation_unsupported_language_alert():
    trans = MultilingualTranslator()
    res = trans.translate("ಪ್ರಾಚೀನ ಹಸ್ತಪ್ರತಿ", src_lang="kn", tgt_lang="en")
    assert res.is_skipped is True
    assert res.translated_text == ""
    assert "Kannada is not pre-trained" in res.warning


def test_translation_mock_execution():
    trans = MultilingualTranslator()
    mock_tokenizer = MagicMock()
    mock_tokenizer.lang_code_to_id = {"en_XX": 250004}
    mock_tokenizer.batch_decode.return_value = ["Translated manuscript sentence."]

    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]

    with patch.object(trans, "load_model"), \
         patch.object(trans, "_tokenizer", mock_tokenizer), \
         patch.object(trans, "_model", mock_model):
        res = trans.translate(
            "यह एक प्राचीन पांडुलिपि है।",
            src_lang="hi",
            tgt_lang="en",
        )
        assert res.is_skipped is False
        assert res.translated_text == "Translated manuscript sentence."
        assert res.src_lang == "hi"
        assert res.tgt_lang == "en"

