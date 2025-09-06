"""
Health check router
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.transcription import transcription_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/ping")
async def ping():
    """Simple ping endpoint that doesn't load the model"""
    return {"status": "pong"}

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_loaded = transcription_service.model is not None
        
        if not model_loaded:
            transcription_service.load_model()
            model_loaded = True
        
        device_info = transcription_service.get_device_info()
            
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            device_info=device_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device_info={},
            error=str(e)
        )
