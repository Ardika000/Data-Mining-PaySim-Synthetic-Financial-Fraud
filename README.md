---
title: PaySim KDD Dashboard
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Data Mining — PaySim Synthetic Financial Fraud (KDD)

Dashboard interaktif (Python Dash) untuk mengomunikasikan temuan proses **Knowledge Discovery in Databases**
atas dataset PaySim (6,3 juta transaksi). Mining dilakukan **tanpa label**; `isFraud` hanya untuk validasi.

**Menu:** Ringkasan · Segmentasi · Pola Normal (aturan JIKA→MAKA) · Deteksi Anomali · Insight Bisnis · Dokumentasi.

## Menjalankan lokal
```bash
cd notebooks/Phase5/dashboard
pip install -r requirements.txt
python app.py        # buka http://localhost:8050
```

## Deploy (Hugging Face Spaces)
Repo ini berisi `Dockerfile` di root. Buat Space baru (SDK **Docker**), lalu push repo ini —
Space otomatis membangun & menjalankan dashboard di port 7860. Data agregat kecil sudah disertakan
di `notebooks/Phase5/data/` (dihasilkan oleh `notebooks/Phase5/phase5_prepare_data.ipynb`).
