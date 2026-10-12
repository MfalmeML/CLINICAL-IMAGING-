# Research prototype. Not for clinical use.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for SimpleITK and pydicom.
# Kept minimal: no build tools in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before source so layer caching works.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn

# Source
COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONPATH=/app/src
EXPOSE 8000

# Run as non-root.
RUN useradd --create-home --uid 1000 imagex
USER imagex

CMD ["uvicorn", "imagex.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
