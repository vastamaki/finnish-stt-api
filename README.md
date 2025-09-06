# Finnish Speech-to-Text API

A FastAPI-based Finnish Speech-to-Text API using the faster-whisper library with NVIDIA GPU support.

This API uses the `Finnish-NLP/whisper-large-finnish-v3-ct2` model for accurate Finnish speech recognition.

## Features

- FastAPI with automatic OpenAPI documentation
- Multiple input formats: WAV files, PCM data, URLs, Opus packets
- GPU acceleration with CUDA support
- Docker containerization with NVIDIA runtime
- Modular code structure with separate routers
- Word-level timestamps and confidence scores
- Health monitoring and API information endpoints

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py        # Health check endpoint
│   │   └── transcribe.py    # Transcription endpoints
│   └── services/
│       ├── __init__.py
│       └── transcription.py # Whisper service logic
├── main.py                  # Server runner
├── test_api.py              # API test script
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
└── server.py                # Legacy Flask server (deprecated)
```

## Installation

### Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
python main.py --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
python main.py --host 0.0.0.0 --port 8000 --reload
```

### Docker

1. Build and run with Docker Compose:

```bash
docker-compose up --build
```

2. Or build and run manually:

```bash
docker build -t finnish-stt-api .
docker run -p 8000:8000 --gpus all finnish-stt-api
```

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

- `GET /health` - Check API health and model status

### Transcription

- `POST /transcribe` - Transcribe uploaded WAV files or PCM data
- `POST /transcribe/raw` - Transcribe raw PCM data from request body
- `POST /transcribe/url` - Transcribe audio from URL
- `POST /transcribe/opus` - Transcribe raw Opus packets from request body

## Usage Examples

### File Upload

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "beam_size=5"
```

### URL Transcription

```bash
curl -X POST "http://localhost:8000/transcribe/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/audio.wav",
    "word_timestamps": true
  }'
```

### PCM Data

```bash
curl -X POST "http://localhost:8000/transcribe/raw" \
  -H "Content-Type: application/octet-stream" \
  -H "X-Sample-Rate: 16000" \
  -H "X-Channels: 1" \
  -H "X-Sample-Width: 2" \
  --data-binary @audio.pcm
```

### Opus Packets

```bash
curl -X POST "http://localhost:8000/transcribe/opus" \
  -H "Content-Type: application/octet-stream" \
  -H "X-Sample-Rate: 48000" \
  -H "X-Channels: 1" \
  --data-binary @audio.opus
```
