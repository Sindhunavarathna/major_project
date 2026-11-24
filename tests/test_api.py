"""
Automated Unit and Integration Tests for FastAPI Server Endpoints.

Tests:
1. Root endpoint (GET /) serves index.html with 200 OK.
2. Status endpoint (GET /api/status) reports model in-memory telemetry and quantization.
3. Samples endpoint (GET /api/samples) returns registered manuscript samples.
4. Prediction endpoint (POST /predict) with sample_id returns structured JSON.
5. Prediction endpoint with image upload returns timing, language, script, and translation.
6. Error handling for empty or malformed requests returns appropriate HTTP 400.
"""

import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture(scope="module")
def client():
    """Create a persistent FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint_serves_html(client):
    """Verify that GET / returns the HTML single-page frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "PolyScribe AI" in response.text


def test_api_status_endpoint(client):
    """Verify that GET /api/status reports online status and loaded models."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "models" in data
    assert "quantization" in data
    assert data["quantization"] == "int8_dynamic_cpu"
    assert "Kannada" in data["supported_languages"]
    assert "German" in data["supported_languages"]


def test_api_samples_endpoint(client):
    """Verify that GET /api/samples returns available project samples."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 5
    sample_ids = [s["id"] for s in data["samples"]]
    assert "kannada" in sample_ids
    assert "german" in sample_ids
    assert "arabic" in sample_ids


def test_predict_with_sample_id(client):
    """Verify POST /predict works with sample_id and returns all required fields."""
    response = client.post(
        "/predict",
        data={
            "sample_id": "kannada",
            "model_mode": "custom",
            "deskew": "true",
            "denoise": "false",
            "binarize": "true",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "detected_language" in data
    assert "detected_script" in data
    assert "confidence" in data
    assert "timing" in data

    # Verify fine-grained per-stage profiling fields
    timing = data["timing"]
    for key in ["preprocess_ms", "script_classifier_ms", "ocr_ms", "language_id_ms", "translation_ms", "total_ms"]:
        assert key in timing
        assert isinstance(timing[key], (int, float))
        assert timing[key] >= 0.0


def test_predict_with_uploaded_file(client):
    """Verify POST /predict works with direct multipart image file upload."""
    # Create small synthetic test patch
    img = np.full((64, 128, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    _, buf = cv2.imencode(".png", img)
    file_bytes = buf.tobytes()

    response = client.post(
        "/predict",
        files={"file": ("test_patch.png", file_bytes, "image/png")},
        data={"model_mode": "custom"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "engine_used" in data
    assert "timing" in data
    assert data["timing"]["total_ms"] > 0


def test_predict_with_base64_camera_input(client):
    """Verify POST /predict works with base64 data string (mobile camera capture)."""
    img = np.full((48, 96, 3), 200, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_str}"

    response = client.post(
        "/predict",
        data={
            "image_base64": data_url,
            "model_mode": "baseline",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "timing" in data


def test_predict_missing_image_returns_400(client):
    """Verify POST /predict without file, base64, or sample_id returns 400 Bad Request."""
    response = client.post("/predict", data={"model_mode": "custom"})
    assert response.status_code == 400
    assert "Must provide an image" in response.json()["detail"]


def test_predict_empty_or_blank_image_returns_could_not_recognize(client):
    """Verify that a blank white image returns clear failure notice without crashing or faking text."""
    blank_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".png", blank_img)
    file_bytes = buf.tobytes()

    response = client.post(
        "/predict",
        files={"file": ("blank.png", file_bytes, "image/png")},
        data={"model_mode": "custom"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    # Verify it does not fabricate text
    assert data["recognized_text"] == ""
    assert data["is_uncertain"] is True
    assert data["detected_language"] == "Uncertain"
    assert "Could not recognize text" in (data.get("warning") or "")


def test_download_translation_pdf(client):
    """Verify POST /api/download generates a valid PDF attachment with ReportLab."""
    response = client.post(
        "/api/download",
        data={
            "title": "Kannada Palm-Leaf Note",
            "detected_language": "Kannada",
            "detected_script": "Kannada",
            "confidence": "0.89",
            "recognized_text": "ಪ್ರಾಚೀನ ಹಸ್ತಪ್ರತಿಗಳು",
            "translated_text": "Ancient manuscripts preserve history.",
            "format": "pdf",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in response.headers["content-disposition"]
    # PDF magic header %PDF
    assert response.content.startswith(b"%PDF")


def test_download_translation_text(client):
    """Verify POST /api/download generates text/plain export."""
    response = client.post(
        "/api/download",
        data={
            "title": "German Diary Note",
            "detected_language": "German",
            "detected_script": "Latin",
            "confidence": "0.94",
            "recognized_text": "Guten Morgen",
            "translated_text": "Good morning",
            "format": "txt",
        },
    )
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "attachment; filename=" in response.headers["content-disposition"]
    assert "German" in response.text
    assert "Guten Morgen" in response.text
