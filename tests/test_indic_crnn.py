"""Unit tests for Phase 3: Shared Indic Recognizer (Kannada, Tamil, Telugu)."""

import json
import os
import numpy as np
import pytest
import torch

from src.indic_ocr.charset import UnifiedIndicCharset
from src.indic_ocr.dataset import (
    IndicCRNNDataset,
    IndicSample,
    generate_seed_indic_samples,
    indic_collate_fn,
    split_samples_by_writer,
)
from src.indic_ocr.metrics import compute_cer, compute_corpus_metrics, compute_wer
from src.indic_ocr.model import IndicCRNN
from src.ocr.indic_crnn_engine import IndicCRNNEngine


def test_unified_indic_charset():
    charset = UnifiedIndicCharset()
    # Blank token must be at index 0
    assert charset.BLANK_INDEX == 0
    assert charset.idx_to_char[0] == "<blank>"

    # Ensure all 3 scripts have characters present
    kn_word = "ಕನ್ನಡ"
    ta_word = "தமிழ்"
    te_word = "తెలుగు"

    kn_enc = charset.encode(kn_word)
    ta_enc = charset.encode(ta_word)
    te_enc = charset.encode(te_word)

    assert len(kn_enc) > 0
    assert len(ta_enc) > 0
    assert len(te_enc) > 0

    # Ensure decoding without duplicates or blanks reconstructs original
    assert charset.decode(kn_enc, collapse_repeats=False, remove_blanks=False) == kn_word
    assert charset.decode(ta_enc, collapse_repeats=False, remove_blanks=False) == ta_word
    assert charset.decode(te_enc, collapse_repeats=False, remove_blanks=False) == te_word


def test_ctc_greedy_decoding_logic():
    charset = UnifiedIndicCharset()
    char_a = charset.encode("ಕ")[0]
    char_b = charset.encode("ರ")[0]

    # Raw sequence with repeats and blank (0) tokens:
    # [0, A, A, 0, B, B, B, 0] -> should collapse to "ಕರ"
    raw_indices = [0, char_a, char_a, 0, char_b, char_b, char_b, 0]
    decoded = charset.decode(raw_indices, collapse_repeats=True, remove_blanks=True)
    assert decoded == "ಕರ"


def test_charset_save_and_load(tmp_path):
    charset = UnifiedIndicCharset()
    save_file = str(tmp_path / "vocab.json")
    charset.save(save_file)
    assert os.path.exists(save_file)

    loaded = UnifiedIndicCharset.load(save_file)
    assert len(loaded) == len(charset)
    assert loaded.BLANK_INDEX == 0


def test_indic_crnn_forward_shape():
    charset = UnifiedIndicCharset()
    model = IndicCRNN(num_classes=len(charset), in_channels=1, lstm_hidden=128)

    # Input: batch of 2, 1 channel, height 32, width 128
    dummy_input = torch.randn(2, 1, 32, 128)
    log_probs = model(dummy_input)

    # Time steps T = W / 4 = 128 / 4 = 32
    assert log_probs.shape == (32, 2, len(charset))


def test_zero_leakage_writer_splitting():
    # Generate multi-writer samples
    samples = generate_seed_indic_samples(writers_per_lang=4, words_per_writer=10, seed=42)
    train_s, val_s, test_s, counts = split_samples_by_writer(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    train_writers = {s.writer_id for s in train_s}
    val_writers = {s.writer_id for s in val_s}
    test_writers = {s.writer_id for s in test_s}

    # Strict zero leakage: test writers must never intersect train writers
    assert len(train_writers.intersection(test_writers)) == 0
    assert len(val_writers.intersection(test_writers)) == 0
    assert len(train_s) > 0
    assert len(test_s) > 0


def test_indic_collate_fn():
    charset = UnifiedIndicCharset()
    sample1 = IndicSample(image_array=np.full((32, 80), 255, dtype=np.uint8), text="ಕನ್ನಡ", language="Kannada", writer_id="w1")
    sample2 = IndicSample(image_array=np.full((32, 100), 255, dtype=np.uint8), text="தமிழ்", language="Tamil", writer_id="w2")

    ds = IndicCRNNDataset([sample1, sample2], charset, target_height=32, max_width=128)
    batch = [ds[0], ds[1]]
    images, targets_concat, input_lengths, target_lengths, texts, langs = indic_collate_fn(batch)

    assert images.shape == (2, 1, 32, 128)
    assert input_lengths.shape == (2,)
    assert input_lengths[0].item() == 128 // 4  # 32 timesteps
    assert target_lengths.shape == (2,)
    assert targets_concat.shape[0] == target_lengths.sum().item()


def test_cer_and_wer_metrics():
    ref = "ನಮಸ್ಕಾರ"
    # Exact match
    assert compute_cer(ref, ref) == 0.0
    assert compute_wer("ಹಲೋ ಲೋಕ", "ಹಲೋ ಲೋಕ") == 0.0

    # Substitution of 1 char
    hyp_sub = "ನಮಸ್ಕಾರಿ"
    cer_sub = compute_cer(ref, hyp_sub)
    assert 0.0 < cer_sub < 0.35

    # Complete mismatch
    cer_mismatch = compute_cer("abc", "xyz")
    assert cer_mismatch == 1.0


def test_indic_crnn_engine_interface():
    engine = IndicCRNNEngine()
    assert engine.name == "Indic-CRNN (Shared Kannada-Tamil-Telugu)"

    dummy_img = np.full((40, 150, 3), 255, dtype=np.uint8)
    # If weights not present, should gracefully report fallback/checkpoint missing without raising exception
    result = engine.recognize(dummy_img, lang="kn")
    assert result.engine == engine.name
    assert isinstance(result.confidence, float)
