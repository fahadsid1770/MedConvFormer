from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "COVID-19 Detection API"
    VERSION: str = "1.0.0"

    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000", "http://localhost:5173"]

    # Models
    CNN_MODEL_PATH: str = "./models/cnn_best.onnx"
    VIT_MODEL_PATH: str = "./models/vit_best.onnx"
    CONFIDENCE_THRESHOLD: float = 0.85

    # Redis (optional - can be disabled)
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    CACHE_TTL: int = 3600  # 1 hour

    # Fallback mode (when models not available) - Default to false to load models
    FALLBACK_MODE: bool = os.getenv("FALLBACK_MODE", "false").lower() == "true"

    # File upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".dcm"}

    # Inference
    BATCH_SIZE: int = 4
    MAX_QUEUE_SIZE: int = 100
    INFERENCE_TIMEOUT: int = 30

    # Monitoring
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()