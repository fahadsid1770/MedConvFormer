import pytest
import json
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient


class TestIntegration:
    """Integration tests for full API flow"""

    def test_full_single_prediction_flow(self, client, sample_image_bytes):
        """Test complete single prediction flow"""
        # Create test image
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        image_data = buffer.getvalue()

        files = {"file": ("xray.png", image_data, "image/png")}

        response = client.post("/api/v1/predict/single", files=files)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        required_fields = [
            "prediction", "confidence", "probabilities",
            "model_used", "cnn_confidence", "inference_time", "cached"
        ]
        for field in required_fields:
            assert field in data

        # Validate data types and ranges
        assert data["prediction"] in ["Normal", "Pneumonia", "COVID-19"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["model_used"] in ["cnn", "hybrid"]
        assert 0.0 <= data["cnn_confidence"] <= 1.0
        assert data["inference_time"] > 0
        assert isinstance(data["cached"], bool)

        # Validate probabilities
        assert isinstance(data["probabilities"], dict)
        assert set(data["probabilities"].keys()) == {"Normal", "Pneumonia", "COVID-19"}
        for prob in data["probabilities"].values():
            assert 0.0 <= prob <= 1.0

        # Probabilities should sum approximately to 1
        total_prob = sum(data["probabilities"].values())
        assert abs(total_prob - 1.0) < 0.01

    def test_full_batch_prediction_flow(self, client):
        """Test complete batch prediction flow"""
        # Create multiple test images
        images_data = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(100 + i*20, 150, 200))
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            images_data.append(("xray_{}.jpg".format(i), buffer.getvalue(), "image/jpeg"))

        files = [("files", img_data) for img_data in images_data]

        response = client.post("/api/v1/predict/batch", files=files)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "predictions" in data
        assert len(data["predictions"]) == 3

        for pred in data["predictions"]:
            # Validate each prediction has required fields
            required_fields = [
                "prediction", "confidence", "probabilities",
                "model_used", "cnn_confidence", "inference_time", "cached"
            ]
            for field in required_fields:
                assert field in pred

            # Validate data types and ranges
            assert pred["prediction"] in ["Normal", "Pneumonia", "COVID-19"]
            assert 0.0 <= pred["confidence"] <= 1.0
            assert pred["model_used"] in ["cnn", "hybrid"]
            assert 0.0 <= pred["cnn_confidence"] <= 1.0
            assert pred["inference_time"] > 0
            assert isinstance(pred["cached"], bool)

    def test_cache_operations_flow(self, client, sample_image_bytes):
        """Test cache operations in full flow"""
        # First, get cache stats
        response = client.get("/api/v1/predict/cache/stats")
        assert response.status_code == 200
        initial_stats = response.json()

        # Make a prediction (should cache result)
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        assert response.status_code == 200

        # Check cache stats again (may have increased)
        response = client.get("/api/v1/predict/cache/stats")
        assert response.status_code == 200
        updated_stats = response.json()

        # Cache stats should be valid
        assert "cached_predictions" in updated_stats
        assert "cache_ttl" in updated_stats

        # Clear cache
        response = client.post("/api/v1/predict/cache/clear")
        assert response.status_code == 200
        clear_result = response.json()
        assert "message" in clear_result
        assert "Cache cleared successfully" in clear_result["message"]

    def test_health_and_metrics_flow(self, client):
        """Test health and metrics endpoints"""
        # Health check
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "version" in health_data
        assert "uptime" in health_data

        # Metrics (if enabled)
        response = client.get("/api/v1/metrics")
        # Metrics might be 200 or 404 depending on ENABLE_METRICS setting
        assert response.status_code in [200, 404]

    def test_error_handling_flow(self, client):
        """Test error handling in full flow"""
        # Test with invalid file
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/v1/predict/single", files=files)
        assert response.status_code == 400

        # Test with corrupted image
        files = {"file": ("test.jpg", b"corrupted data", "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        assert response.status_code == 400

        # Test batch with no files
        response = client.post("/api/v1/predict/batch")
        assert response.status_code == 400

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data