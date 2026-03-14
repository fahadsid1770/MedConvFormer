import pytest
import json
from io import BytesIO
from fastapi import HTTPException
from fastapi.testclient import TestClient


class TestPredictAPI:
    """Unit tests for prediction API endpoints"""

    def test_predict_single_success(self, client, sample_image_bytes):
        """Test successful single prediction"""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}

        response = client.post("/api/v1/predict/single", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ["Normal", "Pneumonia", "COVID-19"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert "probabilities" in data
        assert data["model_used"] in ["cnn", "hybrid"]
        assert isinstance(data["cached"], bool)

    def test_predict_single_no_file(self, client):
        """Test single prediction with no file"""
        response = client.post("/api/v1/predict/single")

        assert response.status_code == 422  # Validation error

    def test_predict_single_invalid_file_type(self, client):
        """Test single prediction with invalid file type"""
        files = {"file": ("test.txt", b"not an image", "text/plain")}

        response = client.post("/api/v1/predict/single", files=files)

        assert response.status_code == 400
        assert "File type not allowed" in response.json()["detail"]

    def test_predict_single_file_too_large(self, client):
        """Test single prediction with file too large"""
        large_file = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large.jpg", large_file, "image/jpeg")}

        response = client.post("/api/v1/predict/single", files=files)

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    def test_predict_single_corrupted_image(self, client):
        """Test single prediction with corrupted image data"""
        corrupted_data = b"not an image file"
        files = {"file": ("corrupted.jpg", corrupted_data, "image/jpeg")}

        response = client.post("/api/v1/predict/single", files=files)

        assert response.status_code == 400
        assert "Invalid image file" in response.json()["detail"]

    def test_predict_batch_success(self, client, sample_image_bytes):
        """Test successful batch prediction"""
        files = [
            ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test2.jpg", sample_image_bytes, "image/jpeg"))
        ]

        response = client.post("/api/v1/predict/batch", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert pred["prediction"] in ["Normal", "Pneumonia", "COVID-19"]
            assert isinstance(pred["cached"], bool)

    def test_predict_batch_empty(self, client):
        """Test batch prediction with no files"""
        response = client.post("/api/v1/predict/batch")

        assert response.status_code == 400
        assert "No files provided" in response.json()["detail"]

    def test_predict_batch_too_many_files(self, client, sample_image_bytes):
        """Test batch prediction with too many files"""
        files = [("files", (f"test{i}.jpg", sample_image_bytes, "image/jpeg")) for i in range(101)]

        response = client.post("/api/v1/predict/batch", files=files)

        assert response.status_code == 400
        assert "Too many files" in response.json()["detail"]

    def test_predict_batch_mixed_valid_invalid(self, client, sample_image_bytes):
        """Test batch prediction with mix of valid and invalid files"""
        files = [
            ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test2.txt", b"not an image", "text/plain"))
        ]

        response = client.post("/api/v1/predict/batch", files=files)

        assert response.status_code == 400
        assert "File type not allowed" in response.json()["detail"]

    def test_get_cache_stats(self, client):
        """Test getting cache statistics"""
        response = client.get("/api/v1/predict/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert "cached_predictions" in data
        assert "cache_ttl" in data
        assert isinstance(data["cached_predictions"], int)
        assert isinstance(data["cache_ttl"], int)

    def test_clear_cache(self, client):
        """Test clearing cache"""
        response = client.post("/api/v1/predict/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Cache cleared successfully" in data["message"]

    def test_predict_single_inference_error(self, client, sample_image_bytes, mock_inference_service):
        """Test single prediction with inference service error"""
        mock_inference_service.predict_single.side_effect = RuntimeError("Inference failed")

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)

        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_predict_batch_inference_error(self, client, sample_image_bytes, mock_inference_service):
        """Test batch prediction with inference service error"""
        mock_inference_service.predict_batch.side_effect = RuntimeError("Batch inference failed")

        files = [("files", ("test.jpg", sample_image_bytes, "image/jpeg"))]
        response = client.post("/api/v1/predict/batch", files=files)

        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]