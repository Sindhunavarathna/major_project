"""Unit tests for Unicode script detection and language identification."""

from src.langid import detect_unicode_script, identify_language


def test_detect_unicode_script():
    # Latin
    s1, conf1, _ = detect_unicode_script("Ancient manuscripts preserve history.")
    assert s1 == "Latin"
    assert conf1 > 0.8

    # Devanagari
    s2, conf2, _ = detect_unicode_script("यह एक प्राचीन पांडुलिपि है।")
    assert s2 == "Devanagari"
    assert conf2 > 0.8

    # Kannada
    s3, conf3, _ = detect_unicode_script("ಪ್ರಾಚೀನ ಹಸ್ತಪ್ರತಿಗಳು ಭಾರತದ ಇತಿಹಾಸವನ್ನು ತಿಳಿಸುತ್ತವೆ.")
    assert s3 == "Kannada"
    assert conf3 > 0.8

    # Tamil
    s4, conf4, _ = detect_unicode_script("பண்டைய தமிழ் ஓலைச்சுவடிகள்.")
    assert s4 == "Tamil"
    assert conf4 > 0.8

    # Arabic
    s5, conf5, _ = detect_unicode_script("المخطوطات العربية القديمة في التاريخ.")
    assert s5 == "Arabic"
    assert conf5 > 0.8


def test_identify_language_english():
    text = "Ancient manuscripts preserve the knowledge and philosophy of centuries."
    res = identify_language(text)
    assert res.language == "en"
    assert res.script == "Latin"
    assert res.is_uncertain is False
    assert res.confidence >= 0.5


def test_identify_language_hindi():
    text = "यह एक प्राचीन पांडुलिपि का नमूना है जो इतिहास को दर्शाता है।"
    res = identify_language(text)
    assert res.language == "hi"
    assert res.script == "Devanagari"
    assert res.is_uncertain is False


def test_identify_language_sanskrit():
    # Verse with Visarga and classic Sanskrit endings
    text = "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः ।"
    res = identify_language(text)
    assert res.language == "sa"
    assert res.script == "Devanagari"
    assert res.is_uncertain is False


def test_identify_language_kannada():
    text = "ಪ್ರಾಚೀನ ಹಸ್ತಪ್ರತಿಗಳು ಭಾರತದ ಇತಿಹಾಸ ಮತ್ತು ಪರಂಪರೆಯನ್ನು ತಿಳಿಸುತ್ತವೆ."
    res = identify_language(text)
    assert res.language == "kn"
    assert res.script == "Kannada"
    assert res.is_uncertain is False


def test_identify_language_junk_is_uncertain():
    # Junk inputs like "# 0" or pure symbols must return Uncertain
    res_hash = identify_language("# 0")
    assert res_hash.is_uncertain is True
    assert res_hash.language == "Uncertain"

    res_short = identify_language("a")
    assert res_short.is_uncertain is True
    assert res_short.language == "Uncertain"

