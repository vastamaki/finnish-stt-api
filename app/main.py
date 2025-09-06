#!/usr/bin/env python3
"""
Finnish Whisper REST API Server
FastAPI implementation for transcribing .wav files or PCM data using Whisper
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.routers import health, transcribe
from app.services.transcription import transcription_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Finnish Whisper API",
    description="REST API for Finnish speech-to-text transcription",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(transcribe.router, tags=["transcription"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Finnish STT API server")
    
    try:
        logger.info("Preloading Whisper model...")
        model = transcription_service.load_model()
        device_info = transcription_service.get_device_info()
        logger.info(f"✅ Whisper model preloaded successfully")
        logger.info(f"Device info: {device_info}")
    except Exception as e:
        logger.error(f"❌ Failed to preload model: {e}")
        logger.info("Server will continue, but model loading will happen on first request")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Finnish STT API server")
    
    # Clean up GPU memory if using CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")
    
    # Clear the model reference
    if hasattr(transcription_service, 'model') and transcription_service.model is not None:
        transcription_service.model = None
        logger.info("Model reference cleared")
    
    logger.info("Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description='Finnish Whisper REST API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
