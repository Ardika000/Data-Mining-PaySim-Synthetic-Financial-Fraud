# Data Mining — PaySim Synthetic Financial Fraud (KDD)

Proyek Knowledge Discovery in Databases atas dataset PaySim (6,3 juta transaksi). Mining dilakukan
**tanpa label**; `isFraud` hanya untuk validasi. Deliverable Fase 5 = dashboard interaktif (Python Dash).

## Menjalankan dashboard
```bash
cd notebooks/Phase5/dashboard
pip install -r requirements.txt
python app.py        # buka http://localhost:8050
```
Data agregat dihasilkan oleh `notebooks/Phase5/phase5_prepare_data.ipynb` (env DataMining).
