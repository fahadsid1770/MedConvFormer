from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
import time
import logging
import psutil
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.config import settings
from app.models.prediction import HealthResponse
from app.services.inference_service import InferenceAPIService

logger = logging.getLogger(__name__)

router = APIRouter()

# Track service start time for uptime
_service_start_time = time.time()

# Initialize inference service to check if models load
_inference_service = None
_service_init_error = None

try:
    _inference_service = InferenceAPIService()
    logger.info("Inference service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize inference service: {e}")
    _service_init_error = str(e)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check the health status of the API service."
)
async def health_check():
    """
    Perform a health check of the service.

    Returns service status, version, and uptime.
    In fallback mode, returns healthy status (for demo purposes).
    """
    try:
        # Check if inference service is available
        if _inference_service is None and not _service_init_error:
            raise HTTPException(status_code=503, detail="Inference service not initialized")

        # Check system resources
        memory = psutil.virtual_memory()
        if memory.percent > 95:  # Critical memory usage
            raise HTTPException(status_code=503, detail="High memory usage")

        uptime = time.time() - _service_start_time

        # Determine status based on fallback mode
        status = "healthy"
        if _inference_service and _inference_service.fallback_mode:
            status = "healthy (fallback mode)"

        return HealthResponse(
            status=status,
            version=settings.VERSION,
            uptime=uptime
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get(
    "/status",
    summary="Get detailed service status",
    description="Get detailed information about the service state including models and cache."
)
async def get_status():
    """
    Get detailed service status including:
    - Model availability
    - Fallback mode status
    - Redis connection status
    - Cache statistics
    """
    try:
        if _inference_service is None:
            return {
                "status": "initializing",
                "error": _service_init_error,
                "version": settings.VERSION
            }
        
        cache_stats = _inference_service.get_cache_stats()
        
        return {
            "status": "running",
            "version": settings.VERSION,
            "uptime": time.time() - _service_start_time,
            "models": {
                "loaded": _inference_service.models_loaded,
                "fallback_mode": _inference_service.fallback_mode,
                "cnn_path": settings.CNN_MODEL_PATH,
                "vit_path": settings.VIT_MODEL_PATH
            },
            "cache": cache_stats,
            "redis": {
                "enabled": _inference_service.redis_enabled,
                "url": settings.REDIS_URL if _inference_service.redis_enabled else None
            },
            "config": {
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "batch_size": settings.BATCH_SIZE
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    description="Expose application metrics in Prometheus format.",
    response_class=PlainTextResponse,
    responses={
        200: {
            "content": {"text/plain": {}},
            "description": "Metrics in Prometheus format"
        }
    }
)
async def metrics():
    """
    Expose Prometheus metrics.

    Returns metrics in Prometheus text format.
    """
    if not settings.ENABLE_METRICS:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    try:
        # Generate latest metrics
        metrics_data = generate_latest()

        return PlainTextResponse(
            content=metrics_data.decode('utf-8'),
            media_type=CONTENT_TYPE_LATEST
        )

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")