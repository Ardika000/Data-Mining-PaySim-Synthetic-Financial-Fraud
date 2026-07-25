# Dockerfile untuk Hugging Face Spaces (SDK: docker) — deploy dashboard Phase 5.
# HF membangun image ini dari root repo; app mendengarkan di port 7860 (default HF).
FROM python:3.11-slim

WORKDIR /app

# 1) Dependensi (ramping: hanya runtime app) — cache layer terpisah agar build cepat
COPY notebooks/Phase5/dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) Kode dashboard + data agregat (relasi ../data dipertahankan)
COPY notebooks/Phase5/dashboard/ ./dashboard/
COPY notebooks/Phase5/data/ ./data/

WORKDIR /app/dashboard
EXPOSE 7860
CMD ["gunicorn", "app:server", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:7860"]
