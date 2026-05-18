FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0

COPY requirements-api.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements-api.txt

COPY . .

CMD uvicorn realesrgan_api:app --host 0.0.0.0 --port $PORT