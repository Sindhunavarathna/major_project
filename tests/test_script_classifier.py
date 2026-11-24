"""Unit tests for PyTorch 5-class script classifier."""

import os
import numpy as np
import pytest
import torch

from src.script_classifier.dataset import (
    ScriptDataset,
    apply_handwriting_augmentations,
    render_script_patch,
)
from src.script_classifier.model import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    ScriptClassifierCNN,
)
from src.script_classifier.predict import ScriptClassifier, predict_script


def test_script_classifier_model_forward():
    model = ScriptClassifierCNN(num_classes=len(CLASS_NAMES), in_channels=1)
    dummy_input = torch.randn(4, 1, 128, 128)
    output = model(dummy_input)

    assert output.shape == (4, len(CLASS_NAMES))
    probs = model.predict_probabilities(dummy_input)
    assert probs.shape == (4, len(CLASS_NAMES))
    sum_probs = probs.sum(dim=1).detach().numpy()
    np.testing.assert_allclose(sum_probs, np.ones(4), atol=1e-4)


def test_render_and_augment_patch():
    patch = render_script_patch("ಕನ್ನಡ", target_size=(128, 128))
    assert patch.shape == (128, 128)
    assert patch.dtype == np.uint8

    augmented = apply_handwriting_augmentations(patch)
    assert augmented.shape == (128, 128)
    assert augmented.dtype == np.uint8


def test_script_dataset():
    p1 = np.full((128, 128), 240, dtype=np.uint8)
    p2 = np.full((128, 128), 250, dtype=np.uint8)
    ds = ScriptDataset([p1, p2], [0, 3], augment=False)

    assert len(ds) == 2
    tensor, label = ds[0]
    assert tensor.shape == (1, 128, 128)
    assert label.item() == 0
    assert tensor.dtype == torch.float32


def test_script_classifier_inference():
    checkpoint_path = "models/script_classifier/best_script_classifier.pth"
    classifier = ScriptClassifier(checkpoint_path=checkpoint_path)

    # Synthetic 128x128 image
    test_img = np.full((128, 128, 3), 240, dtype=np.uint8)
    res = classifier.predict(test_img)

    assert res.script in CLASS_NAMES
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.probabilities) == len(CLASS_NAMES)
    for name in CLASS_NAMES:
        assert name in res.probabilities
    total_prob = sum(res.probabilities.values())
    assert abs(total_prob - 1.0) < 0.05


def test_predict_script_on_sample_images():
    sample_arabic = "sample_data/sample_arabic.png"
    if os.path.exists(sample_arabic):
        res = predict_script(sample_arabic)
        assert res.script in CLASS_NAMES
        assert res.confidence > 0.0
