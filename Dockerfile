FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 10000

CMD ["uvicorn", "realesrgan_api:app", "--host", "0.0.0.0", "--port", "10000"]