"""
Transcription service using faster-whisper
"""

import wave
import tempfile
import logging
import torch
from faster_whisper import WhisperModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self):
        self.model_path = "Finnish-NLP/whisper-large-finnish-v3-ct2"
        self.model = None
        
        self.device = self._get_compatible_device()
        
        self.compute_type_order = self._get_compute_type_order()
        logger.info(f"Using device: {self.device}, compute type order: {self.compute_type_order}")
    
    def _get_compute_type_order(self):
        """Determine optimal compute type order based on device and memory"""
        if self.device == "cpu":
            return ["int8", "float32"]
        
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory_gb:.1f}GB")
            
            if gpu_memory_gb < 6:
                logger.info("Memory-constrained GPU detected, prioritizing int8 compute type")
                return ["int8", "float32", "float16"]
            elif gpu_memory_gb < 8:
                return ["float32", "int8", "float16"]
            else:
                return ["float16", "float32", "int8"]
                
        except Exception as e:
            logger.warning(f"Could not determine GPU memory, using conservative order: {e}")
            return ["int8", "float32", "float16"]
    def _get_compatible_device(self):
        """Determine the best compatible device with robust fallback"""
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return "cpu"
        
        try:
            device_capability = torch.cuda.get_device_capability(0)
            major, minor = device_capability
            compute_capability = major * 10 + minor
            
            logger.info(f"GPU compute capability: sm_{compute_capability}")
            
            if compute_capability < 35:
                logger.warning(f"GPU sm_{compute_capability} is too old for modern PyTorch, using CPU")
                return "cpu"
            
            test_tensor = torch.tensor([1.0], device="cuda")
            _ = test_tensor + 1

            try:
                import torch.nn as nn
                test_conv = nn.Conv1d(1, 1, 1).cuda()
                test_input = torch.randn(1, 1, 10).cuda()
                _ = test_conv(test_input)
                logger.info(f"CUDA and cuDNN test passed for sm_{compute_capability}, using GPU")
                return "cuda"
            except Exception as cudnn_error:
                logger.warning(f"cuDNN test failed: {cudnn_error}")
                logger.warning("CUDA is available but cuDNN has issues, falling back to CPU")
                return "cpu"
            
        except Exception as e:
            logger.warning(f"CUDA compatibility test failed: {e}, falling back to CPU")
            return "cpu"
    
    def load_model(self):
        """Load the Finnish Whisper model with optimized compute type selection"""
        if self.model is None:
            logger.info(f"Loading model: {self.model_path}")
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            model_loaded = False
            last_error = None
            
            for i, compute_type in enumerate(self.compute_type_order):
                try:
                    logger.info(f"Attempting to load model with compute_type: {compute_type} (attempt {i+1}/{len(self.compute_type_order)})")
                    
                    self.model = WhisperModel(
                        self.model_path, 
                        device=self.device, 
                        compute_type=compute_type,
                        download_root=None
                    )
                    
                    self.compute_type = compute_type
                    logger.info(f"✅ Model loaded successfully on {self.device} with compute_type: {compute_type}")
                    model_loaded = True
                    
                    if self.device == "cuda":
                        gpu_props = torch.cuda.get_device_properties(0)
                        total_memory = gpu_props.total_memory / 1024**3
                        allocated = torch.cuda.memory_allocated(0) / 1024**3
                        reserved = torch.cuda.memory_reserved(0) / 1024**3
                        logger.info(f"GPU: {gpu_props.name}, Memory: {total_memory:.1f}GB total, {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
                    break
                    
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    
                    if "out of memory" in error_msg.lower():
                        logger.warning(f"Failed to load model with {compute_type}: Out of GPU memory")
                    elif "float16" in error_msg and "not support" in error_msg:
                        logger.warning(f"Failed to load model with {compute_type}: Float16 not supported on this device")
                    elif "cudnn" in error_msg.lower() or "libcudnn" in error_msg.lower():
                        logger.error(f"cuDNN error detected: {error_msg}")
                        logger.warning("Falling back to CPU due to cuDNN issues")
                        self.device = "cpu"
                        self.compute_type_order = ["int8", "float32"]
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        logger.warning(f"Failed to load model with {compute_type}: {error_msg}")
                    
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    continue
            
            if not model_loaded:
                logger.error(f"❌ Failed to load model with any compute type. Last error: {last_error}")
                raise RuntimeError(f"Could not load Whisper model with any supported compute type. Last error: {last_error}")
                
        return self.model
    
    def transcribe_audio(self, audio_path: str, language: str = "fi", 
                        beam_size: int = 5, word_timestamps: bool = True) -> Dict[str, Any]:
        """Transcribe audio file with optimized memory usage"""
        model = self.load_model()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            if hasattr(self, 'compute_type') and self.compute_type == "int8":
                beam_size = min(beam_size, 3)
                logger.info(f"Using reduced beam_size={beam_size} for int8 compute type")
            
            segments, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1000)
            )
            
            segment_list = []
            full_text = ""
            
            for segment in segments:
                segment_data = {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                }
                
                if hasattr(segment, 'words') and segment.words:
                    segment_data["words"] = [
                        {
                            "start": round(word.start, 2),
                            "end": round(word.end, 2),
                            "word": word.word,
                            "probability": round(word.probability, 3)
                        }
                        for word in segment.words
                    ]
                
                segment_list.append(segment_data)
                full_text += segment.text
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "success": True,
                "text": full_text.strip(),
                "segments": segment_list,
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 2),
                "model_info": {
                    "device": self.device,
                    "compute_type": getattr(self, 'compute_type', 'unknown'),
                    "beam_size": beam_size
                }
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return {
                "success": False,
                "error": str(e)
            }

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        device_info = {
            "device": self.device,
            "compute_type": getattr(self, 'compute_type', 'not_set'),
            "compute_type_order": getattr(self, 'compute_type_order', [])
        }
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            device_info.update({
                "cuda_available": True,
                "gpu_name": gpu_props.name,
                "gpu_memory_total": f"{gpu_props.total_memory / 1024**3:.1f}GB",
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f}GB",
                "gpu_compute_capability": f"sm_{gpu_props.major}{gpu_props.minor}"
            })
        else:
            device_info["cuda_available"] = False
            
        return device_info

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> str:
    """Convert PCM data to temporary WAV file"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    return temp_file.name

transcription_service = TranscriptionService()
