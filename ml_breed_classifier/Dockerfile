# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ml_breed_classifier/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY ml_breed_classifier /app/ml_breed_classifier

EXPOSE 8000
# Bind to Render's provided $PORT if present, else default to 8000
CMD ["sh", "-c", "uvicorn ml_breed_classifier.backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
