# ─────────────────────────────────────────────────────────────────────────────
# Keemography — single-container deploy
# FastAPI (uvicorn) serves both the REST API and the built React frontend.
# Railway injects $PORT; matches /api/health for health-checks.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System packages: ffmpeg + Node 20 ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        git \
        curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so this layer is cached when code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── React frontend: install then build ───────────────────────────────────────
# Copy package files first for layer caching.
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci --prefer-offline

COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# ── App source code ───────────────────────────────────────────────────────────
COPY . .

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000
# Railway provides $PORT; fall back to 8000 locally.
# --timeout-keep-alive: keep connections alive for 5 min (chunked upload sessions)
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300
