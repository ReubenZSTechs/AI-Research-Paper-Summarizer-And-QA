FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

COPY . .

EXPOSE 15000

CMD ['uvicorn', 'backend.api.main:app', "--host", "0.0.0.0", "--port", "15000", "--workers", "1", "--timeout-keep-alive", "1400"]