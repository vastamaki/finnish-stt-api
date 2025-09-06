FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/conda/lib:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/cache
ENV HUGGINGFACE_HUB_CACHE=/app/cache

COPY app/ ./app/
COPY main.py .

EXPOSE 8000

# Use exec form to ensure proper signal handling
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
