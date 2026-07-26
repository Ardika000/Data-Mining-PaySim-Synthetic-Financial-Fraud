# Dockerfile untuk Railway — build dashboard Phase 5 dari root repo.
# Railway menyuntikkan $PORT saat runtime; app harus listen di situ.
FROM python:3.11-slim

WORKDIR /app

# 1) Dependensi (ramping: hanya runtime app)
COPY notebooks/Phase5/dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) Kode dashboard + data agregat (relasi ../data dipertahankan)
COPY notebooks/Phase5/dashboard/ ./dashboard/
COPY notebooks/Phase5/data/ ./data/

WORKDIR /app/dashboard

# shell form agar $PORT ter-expand; fallback 8080 bila tak diset
CMD gunicorn app:server --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8080}
