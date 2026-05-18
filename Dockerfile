FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "realesrgan_api:app", "--host", "0.0.0.0", "--port", "8000"]