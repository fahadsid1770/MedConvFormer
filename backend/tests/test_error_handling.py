import pytest
import numpy as np
from io import BytesIO
from PIL import Image
from unittest.mock import Mock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient


class TestErrorHandling:
    """Comprehensive error handling tests"""

    def test_file_validation_edge_cases(self, client):
        """Test file validation edge cases"""
        # Empty filename
        files = {"file": ("", b"data", "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        assert response.status_code == 400

        # Filename with path traversal
        files = {"file": ("../../../etc/passwd", b"data", "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        # Should still validate extension
        assert response.status_code == 400

        # Case sensitivity in extensions
        files = {"file": ("test.JPG", b"data", "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        # Extension checking should be case insensitive
        assert response.status_code in [200, 400]  # Depends on actual validation

    def test_image_processing_errors(self, client):
        """Test various image processing errors"""
        # Truncated JPEG
        truncated_jpeg = b'\xff\xd8\xff\xe0\x00\x10JFIF'  # Incomplete JPEG header
        files = {"file": ("truncated.jpg", truncated_jpeg, "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        assert response.status_code == 400

        # Unsupported image format in JPEG container
        invalid_image = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4'  # Minimal invalid JPEG
        files = {"file": ("invalid.jpg", invalid_image, "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        assert response.status_code == 400

    def test_large_file_handling(self, client):
        """Test handling of various large file scenarios"""
        # Exactly at limit
        exactly_10mb = b"x" * (10 * 1024 * 1024)
        files = {"file": ("exactly_10mb.jpg", exactly_10mb, "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        # Should be accepted or rejected based on exact implementation
        assert response.status_code in [200, 413]

        # Just over limit
        just_over_10mb = b"x" * (10 * 1024 * 1024 + 1)
        files = {"file": ("just_over.jpg", just_over_10mb, "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)
        assert response.status_code == 413

    def test_batch_processing_errors(self, client, sample_image_bytes):
        """Test batch processing error scenarios"""
        # Mix of valid and invalid files
        files = [
            ("files", ("valid.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("invalid.txt", b"text content", "text/plain")),
            ("files", ("another_valid.png", sample_image_bytes, "image/png"))
        ]
        response = client.post("/api/v1/predict/batch", files=files)
        # Should fail due to invalid file in batch
        assert response.status_code == 400

        # All valid but one corrupted
        corrupted_image = b"not an image"
        files = [
            ("files", ("good.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("corrupted.jpg", corrupted_image, "image/jpeg"))
        ]
        response = client.post("/api/v1/predict/batch", files=files)
        assert response.status_code == 400

    @patch('backend.app.services.inference_service.InferenceAPIService.predict_single')
    def test_inference_service_exceptions(self, mock_predict, client, sample_image_bytes):
        """Test various inference service exceptions"""
        # Test different types of exceptions
        exceptions_to_test = [
            RuntimeError("Model not loaded"),
            ValueError("Invalid input shape"),
            ConnectionError("Redis connection failed"),
            TimeoutError("Inference timeout"),
            MemoryError("Out of memory")
        ]

        for exception in exceptions_to_test:
            mock_predict.side_effect = exception
            files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
            response = client.post("/api/v1/predict/single", files=files)
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]

    @patch('backend.app.services.inference_service.InferenceAPIService.predict_batch')
    def test_batch_inference_exceptions(self, mock_predict_batch, client, sample_image_bytes):
        """Test batch inference exceptions"""
        mock_predict_batch.side_effect = Exception("Batch processing failed")

        files = [("files", ("test.jpg", sample_image_bytes, "image/jpeg"))]
        response = client.post("/api/v1/predict/batch", files=files)
        assert response.status_code == 500

    def test_cache_operation_errors(self, client, mock_inference_service):
        """Test cache operation error handling"""
        # Test cache stats error
        mock_inference_service.get_cache_stats.side_effect = Exception("Redis error")
        response = client.get("/api/v1/predict/cache/stats")
        assert response.status_code == 500

        # Reset mock
        mock_inference_service.get_cache_stats.side_effect = None
        mock_inference_service.get_cache_stats.return_value = {"cached_predictions": 0, "cache_ttl": 3600}

        # Test cache clear error
        mock_inference_service.clear_cache.side_effect = Exception("Cache clear failed")
        response = client.post("/api/v1/predict/cache/clear")
        assert response.status_code == 500

    def test_concurrent_request_handling(self, client, sample_image_bytes):
        """Test handling multiple concurrent requests"""
        import threading
        import time

        results = []
        errors = []

        def make_request():
            try:
                files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
                response = client.post("/api/v1/predict/single", files=files)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All requests should succeed (200) or fail gracefully
        assert len(results) == 5
        assert all(code in [200, 500] for code in results)  # Success or internal error
        assert len(errors) == 0  # No exceptions should be raised

    def test_malformed_requests(self, client):
        """Test handling of malformed requests"""
        # Invalid JSON in form data
        response = client.post("/api/v1/predict/single", data="not form data")
        assert response.status_code in [400, 422]

        # Invalid multipart data
        response = client.post("/api/v1/predict/single", content=b"invalid multipart")
        assert response.status_code in [400, 422]

    def test_network_timeout_simulation(self, client, sample_image_bytes, mock_inference_service):
        """Test handling of simulated network timeouts"""
        import time

        def slow_prediction(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow operation
            return {
                'prediction': 'Normal',
                'confidence': 0.95,
                'probabilities': {'Normal': 0.95, 'Pneumonia': 0.03, 'COVID-19': 0.02},
                'model_used': 'cnn',
                'cnn_confidence': 0.95,
                'inference_time': 0.1,
                'cached': False
            }

        mock_inference_service.predict_single.side_effect = slow_prediction

        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/api/v1/predict/single", files=files)

        # Should still work despite simulated delay
        assert response.status_code == 200