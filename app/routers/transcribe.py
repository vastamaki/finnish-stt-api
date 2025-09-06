"""
Transcription router
"""

from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from app.models.schemas import (
    TranscriptionResponse, 
    URLTranscriptionRequest, 
)
from app.services.transcription import transcription_service, pcm_to_wav
from werkzeug.utils import secure_filename
import tempfile
import os
import logging
import traceback
import urllib.request

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    request: Request,
    file: UploadFile = File(None),
    pcm_data: UploadFile = File(None),
    beam_size: int = Form(5),
    word_timestamps: bool = Form(True),
    sample_rate: int = Form(16000),
    channels: int = Form(1),
    sample_width: int = Form(2)
):
    """Main transcription endpoint for file uploads"""
    try:
        temp_files_to_cleanup = []
        
        try:
            if file is not None:
                if file.filename == '':
                    raise HTTPException(status_code=400, detail="No file selected")
                
                filename = secure_filename(file.filename) if hasattr(file, 'filename') else 'audio_file'
                temp_file = tempfile.NamedTemporaryFile(suffix=f'_{filename}', delete=False)
                
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                
                audio_path = temp_file.name
                temp_files_to_cleanup.append(audio_path)
                
            elif pcm_data is not None:
                content = await pcm_data.read()
                
                audio_path = pcm_to_wav(content, sample_rate, channels, sample_width)
                temp_files_to_cleanup.append(audio_path)
                
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="No audio data provided. Use 'file' for WAV files or 'pcm_data' for PCM data"
                )
            
            result = transcription_service.transcribe_audio(
                audio_path, 
                language='fi',
                beam_size=beam_size,
                word_timestamps=word_timestamps
            )
            
            return TranscriptionResponse(**result)
            
        finally:
            for temp_file in temp_files_to_cleanup:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription endpoint error: {traceback.format_exc()}")
        return TranscriptionResponse(
            success=False,
            error=f"Server error: {str(e)}"
        )

@router.post("/transcribe/url", response_model=TranscriptionResponse)
async def transcribe_from_url(request_data: URLTranscriptionRequest):
    """Transcribe audio from URL"""
    try:
        # Download and transcribe
        temp_file = tempfile.NamedTemporaryFile(suffix='.audio', delete=False)
        
        try:
            urllib.request.urlretrieve(request_data.url, temp_file.name)
            
            result = transcription_service.transcribe_audio(
                temp_file.name,
                language=request_data.language.value,
                beam_size=request_data.beam_size,
                word_timestamps=False
            )
            
            return TranscriptionResponse(**result)
            
        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass
                
    except Exception as e:
        logger.error(f"URL transcription error: {traceback.format_exc()}")
        return TranscriptionResponse(
            success=False,
            error=f"Server error: {str(e)}"
        )
