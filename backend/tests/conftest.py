import pytest
import numpy as np
from io import BytesIO
from PIL import Image
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import json


@pytest.fixture
def mock_config():
    """Mock configuration settings"""
    from unittest.mock import patch
    with patch('backend.app.core.config.settings') as mock_settings:
        mock_settings.ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}
        mock_settings.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        mock_settings.MAX_QUEUE_SIZE = 100
        mock_settings.CONFIDENCE_THRESHOLD = 0.85
        mock_settings.REDIS_URL = "redis://localhost:6379"
        mock_settings.CACHE_TTL = 3600
        mock_settings.BATCH_SIZE = 4
        mock_settings.VERSION = "1.0.0"
        mock_settings.ENABLE_METRICS = True
        yield mock_settings


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing"""
    # Create a simple RGB image
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_image_array():
    """Create sample numpy image array"""
    return np.random.rand(224, 224, 3).astype(np.uint8) * 255


@pytest.fixture
def mock_upload_file(sample_image_bytes):
    """Create a mock UploadFile"""
    upload_file = Mock(spec=UploadFile)
    upload_file.filename = "test_image.jpg"
    upload_file.file = BytesIO(sample_image_bytes)
    upload_file.read.return_value = sample_image_bytes
    return upload_file


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis_mock = Mock()
    redis_mock.get.return_value = None  # No cache hit by default
    redis_mock.setex.return_value = True
    redis_mock.keys.return_value = []
    redis_mock.delete.return_value = 0
    return redis_mock


@pytest.fixture
def mock_inference_service(mock_redis):
    """Mock the ML InferenceService"""
    from unittest.mock import patch, Mock
    instance = Mock()
    instance.predict_single.return_value = {
        'prediction': 'Normal',
        'confidence': 0.95,
        'probabilities': {'Normal': 0.95, 'Pneumonia': 0.03, 'COVID-19': 0.02},
        'model_used': 'cnn',
        'cnn_confidence': 0.95,
        'inference_time': 0.1,
        'cached': False
    }
    instance.predict_batch.return_value = [
        {
            'prediction': 'Normal',
            'confidence': 0.95,
            'probabilities': {'Normal': 0.95, 'Pneumonia': 0.03, 'COVID-19': 0.02},
            'model_used': 'cnn',
            'cnn_confidence': 0.95,
            'inference_time': 0.1,
            'cached': False
        }
    ]
    instance.get_cache_stats.return_value = {
        'cached_predictions': 5,
        'cache_ttl': 3600
    }
    instance.clear_cache.return_value = {"message": "Cache cleared successfully"}

    # Patch the InferenceService class in the inference_service module
    with patch('backend.ml.inference.inference_service.InferenceService', return_value=instance):
        yield instance


@pytest.fixture
def client(mock_config, mock_inference_service):
    """FastAPI test client"""
    from backend.app.main import app
    return TestClient(app)


@pytest.fixture
def mock_ort_session():
    """Mock ONNX runtime session"""
    session = Mock()
    session.get_inputs.return_value = [Mock(name="input")]
    session.run.return_value = [np.array([[0.1, 0.8, 0.1]])]  # Mock logits
    return session


@pytest.fixture
def mock_preprocessor():
    """Mock XRayPreprocessor"""
    prep = Mock()
    # Mock tensor with numpy method
    mock_tensor = Mock()
    mock_tensor.numpy.return_value = np.random.rand(1, 3, 224, 224).astype(np.float32)
    prep.preprocess_single.return_value = mock_tensor
    return prep