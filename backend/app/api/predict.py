from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import logging

from app.models.prediction import PredictionResponse, BatchPredictionResponse, ErrorResponse
from app.services.inference_service import InferenceAPIService

logger = logging.getLogger(__name__)

router = APIRouter()
inference_service = InferenceAPIService()


@router.post(
    "/single",
    response_model=PredictionResponse,
    summary="Predict single X-ray image",
    description="Upload a single X-ray image and get prediction for COVID-19, Pneumonia, or Normal."
)
async def predict_single(file: UploadFile = File(..., description="X-ray image file")):
    """
    Perform prediction on a single uploaded X-ray image.

    - **file**: Image file (JPG, PNG, JPEG, DCM)
    - Returns prediction result with confidence scores
    """
    try:
        result = await inference_service.predict_single(file)
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in single prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict batch of X-ray images",
    description="Upload multiple X-ray images and get batch predictions."
)
async def predict_batch(files: List[UploadFile] = File(..., description="List of X-ray image files")):
    """
    Perform batch prediction on multiple uploaded X-ray images.

    - **files**: List of image files (JPG, PNG, JPEG, DCM)
    - Returns list of prediction results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        results = await inference_service.predict_batch(files)
        return BatchPredictionResponse(predictions=[PredictionResponse(**r) for r in results])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/cache/stats",
    summary="Get cache statistics",
    description="Retrieve statistics about cached predictions."
)
async def get_cache_stats():
    """Get cache statistics"""
    try:
        return inference_service.get_cache_stats()
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache stats")


@router.post(
    "/cache/clear",
    summary="Clear prediction cache",
    description="Clear all cached prediction results."
)
async def clear_cache():
    """Clear the prediction cache"""
    try:
        return inference_service.clear_cache()
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")