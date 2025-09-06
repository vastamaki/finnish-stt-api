"""
Pydantic models for request and response schemas
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class WordTimestamp(BaseModel):
    """Word-level timestamp information"""
    start: float
    end: float
    word: str
    probability: float

class Segment(BaseModel):
    """Transcription segment with timing"""
    start: float
    end: float
    text: str
    words: Optional[List[WordTimestamp]] = None

class TranscriptionRequest(BaseModel):
    """Base transcription request parameters"""
    language: str = "fi"
    beam_size: int = 5
    word_timestamps: bool = True

class URLTranscriptionRequest(TranscriptionRequest):
    """URL transcription request"""
    url: str

class PCMParameters(BaseModel):
    """PCM audio parameters"""
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2

class OpusParameters(BaseModel):
    """Opus audio parameters"""
    sample_rate: int = 48000  # Opus standard sample rates: 48000, 24000, 16000, 12000, 8000
    channels: int = 1

class TranscriptionResponse(BaseModel):
    """Transcription response"""
    success: bool
    text: Optional[str] = None
    segments: Optional[List[Segment]] = None
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device_info: Dict[str, Any]
    error: Optional[str] = None

class InfoResponse(BaseModel):
    """API information response"""
    name: str
    model: str
    supported_formats: List[str]
    endpoints: Dict[str, str]
    parameters: Dict[str, str]
