import numpy as np
import logging
from typing import List
from PIL import Image
import io
from fastapi import UploadFile, HTTPException

from ml.inference.inference_service import InferenceService
from app.core.config import settings

logger = logging.getLogger(__name__)


class InferenceAPIService:
    """
    API wrapper for the InferenceService, handling file uploads and image processing.
    """

    def __init__(self):
        self.inference_service = InferenceService()

    @property
    def fallback_mode(self) -> bool:
        """Check if running in fallback mode"""
        return self.inference_service.fallback_mode
    
    @property
    def models_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.inference_service.models_loaded
    
    @property
    def redis_enabled(self) -> bool:
        """Check if Redis is enabled"""
        return self.inference_service.redis_enabled

    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file extension
        file_ext = file.filename.lower().split('.')[-1]
        if f".{file_ext}" not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )

        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Seek back to beginning

        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE} bytes"
            )

    def _file_to_numpy(self, file: UploadFile) -> np.ndarray:
        """Convert uploaded file to numpy array"""
        try:
            contents = file.file.read()
            image = Image.open(io.BytesIO(contents))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to numpy array
            image_array = np.array(image)

            return image_array

        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file")

    async def predict_single(self, file: UploadFile) -> dict:
        """Perform single prediction from uploaded file"""
        self._validate_file(file)
        image_array = self._file_to_numpy(file)

        try:
            result = self.inference_service.predict_single(image_array)
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

    async def predict_batch(self, files: List[UploadFile]) -> List[dict]:
        """Perform batch prediction from multiple uploaded files"""
        if len(files) > settings.MAX_QUEUE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Max: {settings.MAX_QUEUE_SIZE}"
            )

        image_arrays = []
        for file in files:
            self._validate_file(file)
            image_array = self._file_to_numpy(file)
            image_arrays.append(image_array)

        try:
            results = self.inference_service.predict_batch(image_arrays)
            return results
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Batch prediction failed")

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return self.inference_service.get_cache_stats()

    def clear_cache(self) -> dict:
        """Clear cache"""
        self.inference_service.clear_cache()
        return {"message": "Cache cleared successfully"}