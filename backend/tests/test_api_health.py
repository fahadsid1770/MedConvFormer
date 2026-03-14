import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


class TestHealthAPI:
    """Unit tests for health API endpoints"""

    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime" in data
        assert isinstance(data["uptime"], float)
        assert data["uptime"] >= 0

    @patch('backend.app.api.health._inference_service', None)
    def test_health_check_inference_service_unavailable(self, client):
        """Test health check when inference service is not available"""
        response = client.get("/api/v1/health")

        assert response.status_code == 503
        assert "Inference service not available" in response.json()["detail"]

    @patch('backend.app.api.health.psutil.virtual_memory')
    def test_health_check_high_memory_usage(self, mock_memory, client):
        """Test health check with high memory usage"""
        from unittest.mock import Mock
        # Mock high memory usage
        mock_mem = Mock()
        mock_mem.percent = 96  # Above 95% threshold
        mock_memory.return_value = mock_mem

        response = client.get("/api/v1/health")

        assert response.status_code == 503
        assert "High memory usage" in response.json()["detail"]

    @patch('backend.app.api.health._inference_service')
    def test_health_check_inference_service_exception(self, mock_inference_service, client):
        """Test health check when inference service raises exception"""
        mock_inference_service.side_effect = Exception("Service error")

        response = client.get("/api/v1/health")

        assert response.status_code == 503
        assert "Service unhealthy" in response.json()["detail"]

    def test_metrics_enabled(self, client):
        """Test metrics endpoint when enabled"""
        response = client.get("/api/v1/metrics")

        assert response.status_code == 200
        # Should return Prometheus format text
        assert "text/plain" in response.headers.get("content-type", "")
        content = response.text
        assert len(content) > 0

    @patch('backend.app.api.health.settings.ENABLE_METRICS', False)
    def test_metrics_disabled(self, client):
        """Test metrics endpoint when disabled"""
        response = client.get("/api/v1/metrics")

        assert response.status_code == 404
        assert "Metrics not enabled" in response.json()["detail"]

    @patch('backend.app.api.health.generate_latest')
    def test_metrics_generation_error(self, mock_generate, client):
        """Test metrics endpoint with generation error"""
        mock_generate.side_effect = Exception("Metrics generation failed")

        response = client.get("/api/v1/metrics")

        assert response.status_code == 500
        assert "Failed to generate metrics" in response.json()["detail"]