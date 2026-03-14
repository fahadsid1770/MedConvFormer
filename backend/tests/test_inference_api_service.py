import pytest
import numpy as np
from io import BytesIO
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException, UploadFile

from backend.app.services.inference_service import InferenceAPIService


class TestInferenceAPIService:
    """Unit tests for InferenceAPIService"""

    @pytest.fixture
    def service(self, mock_inference_service):
        """Create InferenceAPIService instance"""
        return InferenceAPIService()

    def test_initialization(self, mock_inference_service):
        """Test service initialization"""
        service = InferenceAPIService()
        assert service.inference_service is not None

    def test_validate_file_valid(self, service, mock_upload_file, mock_config):
        """Test file validation with valid file"""
        # Should not raise exception
        service._validate_file(mock_upload_file)

    def test_validate_file_no_filename(self, service, mock_config):
        """Test file validation with no filename"""
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = None

        with pytest.raises(HTTPException) as exc_info:
            service._validate_file(upload_file)

        assert exc_info.value.status_code == 400
        assert "No file provided" in str(exc_info.value.detail)

    def test_validate_file_invalid_extension(self, service, mock_config):
        """Test file validation with invalid extension"""
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.txt"

        with pytest.raises(HTTPException) as exc_info:
            service._validate_file(upload_file)

        assert exc_info.value.status_code == 400
        assert "File type not allowed" in str(exc_info.value.detail)

    def test_validate_file_too_large(self, service, mock_config):
        """Test file validation with file too large"""
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.jpg"
        upload_file.file = Mock()
        upload_file.file.tell.return_value = 20 * 1024 * 1024  # 20MB

        with pytest.raises(HTTPException) as exc_info:
            service._validate_file(upload_file)

        assert exc_info.value.status_code == 413
        assert "File too large" in str(exc_info.value.detail)

    @patch('backend.app.services.inference_service.Image.open')
    def test_file_to_numpy_success(self, mock_image_open, service, mock_upload_file):
        """Test successful image to numpy conversion"""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.mode = 'RGB'
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img

        # Mock numpy array
        expected_array = np.random.rand(224, 224, 3).astype(np.uint8)
        with patch('backend.app.services.inference_service.np.array', return_value=expected_array):
            result = service._file_to_numpy(mock_upload_file)

        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)
        mock_img.convert.assert_called_once_with('RGB')

    @patch('backend.app.services.inference_service.Image.open')
    def test_file_to_numpy_invalid_image(self, mock_image_open, service, mock_upload_file):
        """Test image conversion with invalid image file"""
        mock_image_open.side_effect = Exception("Invalid image")

        with pytest.raises(HTTPException) as exc_info:
            service._file_to_numpy(mock_upload_file)

        assert exc_info.value.status_code == 400
        assert "Invalid image file" in str(exc_info.value.detail)

    @patch('backend.app.services.inference_service.Image.open')
    def test_file_to_numpy_grayscale_conversion(self, mock_image_open, service, mock_upload_file):
        """Test image conversion with grayscale image"""
        # Mock grayscale image
        mock_img = Mock()
        mock_img.mode = 'L'  # Grayscale
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img

        expected_array = np.random.rand(224, 224, 3).astype(np.uint8)
        with patch('backend.app.services.inference_service.np.array', return_value=expected_array):
            result = service._file_to_numpy(mock_upload_file)

        mock_img.convert.assert_called_once_with('RGB')

    @pytest.mark.asyncio
    async def test_predict_single_success(self, service, mock_upload_file, mock_inference_service):
        """Test successful single prediction"""
        result = await service.predict_single(mock_upload_file)

        assert result['prediction'] == 'Normal'
        assert result['confidence'] == 0.95
        assert result['model_used'] == 'cnn'
        assert not result['cached']

        # Verify validation and conversion were called
        mock_inference_service.predict_single.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_single_validation_error(self, service, mock_inference_service):
        """Test single prediction with validation error"""
        upload_file = Mock(spec=UploadFile)
        upload_file.filename = "test.txt"  # Invalid extension

        with pytest.raises(HTTPException) as exc_info:
            await service.predict_single(upload_file)

        assert exc_info.value.status_code == 400
        mock_inference_service.predict_single.assert_not_called()

    @pytest.mark.asyncio
    async def test_predict_single_inference_error(self, service, mock_upload_file, mock_inference_service):
        """Test single prediction with inference error"""
        mock_inference_service.predict_single.side_effect = RuntimeError("Inference failed")

        with pytest.raises(HTTPException) as exc_info:
            await service.predict_single(mock_upload_file)

        assert exc_info.value.status_code == 500
        assert "Prediction failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, service, mock_inference_service):
        """Test successful batch prediction"""
        upload_files = [Mock(spec=UploadFile) for _ in range(2)]
        for i, uf in enumerate(upload_files):
            uf.filename = f"test_{i}.jpg"
            uf.file = Mock()
            uf.file.tell.return_value = 1024  # Small file

        result = await service.predict_batch(upload_files)

        assert len(result) == 2
        assert all(r['prediction'] == 'Normal' for r in result)
        mock_inference_service.predict_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_batch_empty_files(self, service):
        """Test batch prediction with empty file list"""
        with pytest.raises(HTTPException) as exc_info:
            await service.predict_batch([])

        assert exc_info.value.status_code == 400
        assert "No files provided" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_predict_batch_too_many_files(self, service, mock_config):
        """Test batch prediction with too many files"""
        upload_files = [Mock(spec=UploadFile) for _ in range(101)]  # More than MAX_QUEUE_SIZE
        for uf in upload_files:
            uf.filename = "test.jpg"

        with pytest.raises(HTTPException) as exc_info:
            await service.predict_batch(upload_files)

        assert exc_info.value.status_code == 400
        assert "Too many files" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_predict_batch_validation_error(self, service, mock_inference_service):
        """Test batch prediction with validation error on second file"""
        upload_files = [Mock(spec=UploadFile) for _ in range(2)]
        upload_files[0].filename = "test1.jpg"
        upload_files[0].file = Mock()
        upload_files[0].file.tell.return_value = 1024

        upload_files[1].filename = "test2.txt"  # Invalid extension

        with pytest.raises(HTTPException) as exc_info:
            await service.predict_batch(upload_files)

        assert exc_info.value.status_code == 400
        mock_inference_service.predict_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_predict_batch_inference_error(self, service, mock_inference_service):
        """Test batch prediction with inference error"""
        upload_files = [Mock(spec=UploadFile)]
        upload_files[0].filename = "test.jpg"
        upload_files[0].file = Mock()
        upload_files[0].file.tell.return_value = 1024

        mock_inference_service.predict_batch.side_effect = RuntimeError("Batch inference failed")

        with pytest.raises(HTTPException) as exc_info:
            await service.predict_batch(upload_files)

        assert exc_info.value.status_code == 500
        assert "Batch prediction failed" in str(exc_info.value.detail)

    def test_get_cache_stats(self, service, mock_inference_service):
        """Test getting cache statistics"""
        result = service.get_cache_stats()

        assert result['cached_predictions'] == 5
        assert result['cache_ttl'] == 3600
        mock_inference_service.get_cache_stats.assert_called_once()

    def test_clear_cache(self, service, mock_inference_service):
        """Test clearing cache"""
        result = service.clear_cache()

        assert result['message'] == "Cache cleared successfully"
        mock_inference_service.clear_cache.assert_called_once()