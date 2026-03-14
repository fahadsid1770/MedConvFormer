import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from .inference_service import InferenceService


class TestInferenceService:
    """Test cases for InferenceService"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration settings"""
        with patch('backend.ml.inference.inference_service.settings') as mock_settings:
            mock_settings.CNN_MODEL_PATH = "/fake/cnn_model.onnx"
            mock_settings.VIT_MODEL_PATH = "/fake/vit_model.onnx"
            mock_settings.CONFIDENCE_THRESHOLD = 0.85
            mock_settings.REDIS_URL = "redis://localhost:6379"
            mock_settings.CACHE_TTL = 3600
            mock_settings.BATCH_SIZE = 4
            yield mock_settings

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        return np.random.rand(224, 224, 3).astype(np.uint8) * 255

    @pytest.fixture
    def mock_ort_session(self):
        """Mock ONNX runtime session"""
        session = Mock()
        session.get_inputs.return_value = [Mock(name="input")]
        session.run.return_value = [np.array([[0.1, 0.8, 0.1]])]  # Mock logits
        return session

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_initialization_success(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config):
        """Test successful initialization"""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_ort.return_value = Mock()

        # Initialize service
        service = InferenceService()

        # Assertions
        assert service.class_names == ['Normal', 'Pneumonia', 'COVID-19']
        assert service.confidence_threshold == 0.85
        mock_ort.assert_called()
        mock_redis.assert_called_once_with("redis://localhost:6379")

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_model_loading_failure(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config):
        """Test model loading failure"""
        mock_path.return_value.exists.return_value = False

        with pytest.raises(FileNotFoundError):
            InferenceService()

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_cache_key_generation(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config, sample_image):
        """Test cache key generation"""
        mock_path.return_value.exists.return_value = True
        mock_ort.return_value = Mock()

        service = InferenceService()
        cache_key = service._get_cache_key(sample_image)

        assert cache_key.startswith("inference:")
        assert len(cache_key) == len("inference:") + 32  # MD5 hash length

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_single_prediction_cnn_only(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config, sample_image):
        """Test single prediction using only CNN (high confidence)"""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_cnn_session = Mock()
        mock_vit_session = Mock()

        # Mock CNN session - high confidence
        mock_cnn_session.get_inputs.return_value = [Mock(name="input")]
        mock_cnn_session.run.return_value = [np.array([[0.05, 0.9, 0.05]])]  # High confidence for Pneumonia

        # Mock ViT session (should not be called)
        mock_vit_session.get_inputs.return_value = [Mock(name="input")]
        mock_vit_session.run.return_value = [np.array([[0.1, 0.8, 0.1]])]

        mock_ort.side_effect = [mock_cnn_session, mock_vit_session]

        # Mock preprocessor
        mock_prep_instance = Mock()
        mock_prep_instance.preprocess_single.return_value.numpy.return_value = np.random.rand(1, 3, 224, 224).astype(np.float32)
        mock_preprocessor.return_value = mock_prep_instance

        # Mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.get.return_value = None  # No cache hit
        mock_redis.return_value = mock_redis_instance

        service = InferenceService()
        result = service.predict_single(sample_image)

        # Assertions
        assert result['prediction'] == 'Pneumonia'
        assert result['model_used'] == 'cnn'
        assert result['confidence'] > 0.85
        assert not result['cached']
        assert 'probabilities' in result
        assert 'inference_time' in result

        # Verify ViT was not called (high CNN confidence)
        mock_vit_session.run.assert_not_called()

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_single_prediction_hybrid(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config, sample_image):
        """Test single prediction using hybrid CNN-ViT (low confidence)"""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_cnn_session = Mock()
        mock_vit_session = Mock()

        # Mock CNN session - low confidence
        mock_cnn_session.get_inputs.return_value = [Mock(name="input")]
        mock_cnn_session.run.return_value = [np.array([[0.4, 0.3, 0.3]])]  # Low confidence

        # Mock ViT session
        mock_vit_session.get_inputs.return_value = [Mock(name="input")]
        mock_vit_session.run.return_value = [np.array([[0.1, 0.8, 0.1]])]  # High confidence for Pneumonia

        mock_ort.side_effect = [mock_cnn_session, mock_vit_session]

        # Mock preprocessor
        mock_prep_instance = Mock()
        mock_prep_instance.preprocess_single.return_value.numpy.return_value = np.random.rand(1, 3, 224, 224).astype(np.float32)
        mock_preprocessor.return_value = mock_prep_instance

        # Mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.get.return_value = None  # No cache hit
        mock_redis.return_value = mock_redis_instance

        service = InferenceService()
        result = service.predict_single(sample_image)

        # Assertions
        assert result['prediction'] == 'Pneumonia'
        assert result['model_used'] == 'hybrid'
        assert not result['cached']

        # Verify both models were called
        mock_cnn_session.run.assert_called()
        mock_vit_session.run.assert_called()

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_cache_hit(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config, sample_image):
        """Test cache hit scenario"""
        mock_path.return_value.exists.return_value = True
        mock_ort.return_value = Mock()

        # Mock Redis with cache hit
        mock_redis_instance = Mock()
        cached_result = {
            'prediction': 'COVID-19',
            'confidence': 0.95,
            'model_used': 'cnn'
        }
        mock_redis_instance.get.return_value = json.dumps(cached_result)
        mock_redis.return_value = mock_redis_instance

        service = InferenceService()
        result = service.predict_single(sample_image)

        # Assertions
        assert result['prediction'] == 'COVID-19'
        assert result['cached'] is True

        # Verify no model inference occurred
        mock_ort.return_value.run.assert_not_called()

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_batch_prediction(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config):
        """Test batch prediction"""
        mock_path.return_value.exists.return_value = True
        mock_cnn_session = Mock()
        mock_vit_session = Mock()

        # Mock sessions
        mock_cnn_session.get_inputs.return_value = [Mock(name="input")]
        mock_cnn_session.run.return_value = [np.array([[0.1, 0.85, 0.05], [0.4, 0.3, 0.3]])]  # Batch of 2

        mock_vit_session.get_inputs.return_value = [Mock(name="input")]
        mock_vit_session.run.return_value = [np.array([[0.1, 0.8, 0.1]])]  # Only for uncertain image

        mock_ort.side_effect = [mock_cnn_session, mock_vit_session]

        # Mock preprocessor
        mock_prep_instance = Mock()
        mock_prep_instance.preprocess_single.return_value.numpy.return_value = np.random.rand(1, 3, 224, 224).astype(np.float32)
        mock_preprocessor.return_value = mock_prep_instance

        # Mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.get.return_value = None  # No cache hits
        mock_redis.return_value = mock_redis_instance

        service = InferenceService()

        # Create batch of images
        images = [
            np.random.rand(224, 224, 3).astype(np.uint8) * 255,
            np.random.rand(224, 224, 3).astype(np.uint8) * 255
        ]

        results = service.predict_batch(images, batch_size=2)

        # Assertions
        assert len(results) == 2
        assert all('prediction' in result for result in results)
        assert all('model_used' in result for result in results)
        assert results[0]['model_used'] == 'cnn'  # High confidence
        assert results[1]['model_used'] == 'hybrid'  # Low confidence

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_batch_prediction_with_cache(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config):
        """Test batch prediction with mixed cache hits and misses"""
        mock_path.return_value.exists.return_value = True
        mock_cnn_session = Mock()
        mock_vit_session = Mock()

        # Mock sessions
        mock_cnn_session.get_inputs.return_value = [Mock(name="input")]
        mock_cnn_session.run.return_value = [np.array([[0.1, 0.85, 0.05]])]  # Only for uncached image

        mock_vit_session.get_inputs.return_value = [Mock(name="input")]
        mock_vit_session.run.return_value = [np.array([[0.1, 0.8, 0.1]])]

        mock_ort.side_effect = [mock_cnn_session, mock_vit_session]

        # Mock preprocessor
        mock_prep_instance = Mock()
        mock_prep_instance.preprocess_single.return_value.numpy.return_value = np.random.rand(1, 3, 224, 224).astype(np.float32)
        mock_preprocessor.return_value = mock_prep_instance

        # Mock Redis - first image cached, second not
        mock_redis_instance = Mock()
        cached_result = {
            'prediction': 'COVID-19',
            'confidence': 0.95,
            'probabilities': {'Normal': 0.02, 'Pneumonia': 0.03, 'COVID-19': 0.95},
            'model_used': 'cnn',
            'cnn_confidence': 0.95,
            'inference_time': 0.05,
            'cached': True
        }
        def mock_get(key):
            if "cached" in key:
                return json.dumps(cached_result)
            return None
        mock_redis_instance.get.side_effect = mock_get
        mock_redis.return_value = mock_redis_instance

        service = InferenceService()

        # Create batch of images
        images = [
            np.random.rand(224, 224, 3).astype(np.uint8) * 255,  # Will be cached
            np.random.rand(224, 224, 3).astype(np.uint8) * 255   # Will be processed
        ]

        results = service.predict_batch(images, batch_size=2)

        # Assertions
        assert len(results) == 2
        assert results[0]['cached'] is True
        assert results[0]['prediction'] == 'COVID-19'
        assert results[1]['cached'] is False
        assert results[1]['prediction'] == 'Pneumonia'

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_cache_operations(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config):
        """Test cache clear and stats operations"""
        mock_path.return_value.exists.return_value = True
        mock_ort.return_value = Mock()

        mock_redis_instance = Mock()
        mock_redis_instance.keys.return_value = ["inference:key1", "inference:key2"]
        mock_redis.return_value = mock_redis_instance

        service = InferenceService()

        # Test cache stats
        stats = service.get_cache_stats()
        assert stats['cached_predictions'] == 2
        assert stats['cache_ttl'] == 3600

        # Test cache clear
        service.clear_cache()
        mock_redis_instance.delete.assert_called_once_with(["inference:key1", "inference:key2"])

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_redis_connection_error(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config, sample_image):
        """Test handling of Redis connection errors"""
        mock_path.return_value.exists.return_value = True
        mock_ort.return_value = Mock()

        mock_redis_instance = Mock()
        mock_redis_instance.get.side_effect = Exception("Redis connection failed")
        mock_redis_instance.setex.side_effect = Exception("Redis connection failed")
        mock_redis.return_value = mock_redis_instance

        # Mock preprocessor
        mock_prep_instance = Mock()
        mock_prep_instance.preprocess_single.return_value.numpy.return_value = np.random.rand(1, 3, 224, 224).astype(np.float32)
        mock_preprocessor.return_value = mock_prep_instance

        service = InferenceService()

        # Should still work despite Redis errors
        result = service.predict_single(sample_image)
        assert 'prediction' in result
        assert result['cached'] is False

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_inference_error_handling(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config, sample_image):
        """Test handling of inference errors"""
        mock_path.return_value.exists.return_value = True

        mock_cnn_session = Mock()
        mock_cnn_session.get_inputs.return_value = [Mock(name="input")]
        mock_cnn_session.run.side_effect = Exception("ONNX runtime error")

        mock_ort.return_value = mock_cnn_session

        # Mock preprocessor
        mock_prep_instance = Mock()
        mock_prep_instance.preprocess_single.return_value.numpy.return_value = np.random.rand(1, 3, 224, 224).astype(np.float32)
        mock_preprocessor.return_value = mock_prep_instance

        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance

        service = InferenceService()

        with pytest.raises(RuntimeError, match="Inference failed"):
            service.predict_single(sample_image)

    @patch('backend.ml.inference.inference_service.ort.InferenceSession')
    @patch('backend.ml.inference.inference_service.redis.from_url')
    @patch('backend.ml.inference.inference_service.XRayPreprocessor')
    @patch('backend.ml.inference.inference_service.Path')
    def test_preprocessing_error(self, mock_path, mock_preprocessor, mock_redis, mock_ort, mock_config, sample_image):
        """Test handling of preprocessing errors"""
        mock_path.return_value.exists.return_value = True
        mock_ort.return_value = Mock()

        # Mock preprocessor to raise error
        mock_prep_instance = Mock()
        mock_prep_instance.preprocess_single.side_effect = Exception("Preprocessing failed")
        mock_preprocessor.return_value = mock_prep_instance

        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance

        service = InferenceService()

        with pytest.raises(RuntimeError, match="Inference failed"):
            service.predict_single(sample_image)


if __name__ == "__main__":
    pytest.main([__file__])