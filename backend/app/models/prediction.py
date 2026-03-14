from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from fastapi import UploadFile


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: str = Field(..., description="Predicted class: Normal, Pneumonia, or COVID-19")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the prediction")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    model_used: str = Field(..., description="Model used: cnn or hybrid")
    cnn_confidence: float = Field(..., ge=0.0, le=1.0, description="CNN confidence score")
    inference_time: float = Field(..., gt=0.0, description="Time taken for inference in seconds")
    cached: bool = Field(..., description="Whether the result was retrieved from cache")
    fallback: Optional[bool] = Field(False, description="Whether fallback mode was used")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status: healthy or unhealthy")
    version: str = Field(..., description="API version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")