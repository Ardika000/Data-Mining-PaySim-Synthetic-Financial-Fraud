# 📘 Phase 3 — Penjelasan Lengkap (Association Rule Mining & Anomaly Pattern Validation)

> Dokumen ini menjelaskan Phase 3 **dari nol** — untuk orang yang belum pernah lihat kode maupun mendengar kata "Apriori". Isi: struktur, logika urutan berpikir, input/proses/output tiap sel, justifikasi pemilihan metode & parameter, dan insight yang dihasilkan.

---

## 🛒 0. Analogi pembuka (mulai dari sini)

Bayangkan kasir supermarket menganalisis ribuan struk. Ia menemukan: *"Orang yang beli roti + selai, 80% juga beli mentega."* Itulah **association rule** — pola "barang apa sering dibeli bersama".

**Phase 3 melakukan hal yang sama, tapi struk diganti transaksi keuangan.** Alih-alih "roti + selai", kita cari **"TRANSFER + menguras saldo + jam kerja sering muncul bersama nominal besar"**. Tujuannya: menemukan hubungan tersembunyi yang **tak terlihat dari tabel/rata-rata biasa**.

---

## 📖 1. Kamus istilah (wajib paham dulu)

| Istilah | Analogi supermarket | Arti di Phase 3 |
|---|---|---|
| **Item** | 1 barang (roti) | 1 nilai atribut, mis. `type=PAYMENT` |
| **Transaksi / basket** | 1 struk | 1 transaksi + label-labelnya |
| **Itemset** | isi 1 struk | kombinasi item |
| **Frequent itemset** | kombinasi yang **sering** | itemset yang muncul di ≥ ambang tertentu |
| **Association rule** | "roti → mentega" | **JIKA** (antecedent) **MAKA** (consequent) |
| **Apriori** | metode kasir cari kombinasi | algoritma pencari frequent itemset yang efisien |

**Tiga ukuran penentu:**

| Ukuran | Pertanyaan | Rumus sederhana |
|---|---|---|
| **Support** | Seberapa **umum** pola ini? | % transaksi yang punya A **dan** B |
| **Confidence** | Seberapa **andal**? | dari yang punya A, berapa % juga punya B |
| **Lift** | Seberapa **kuat vs kebetulan**? | confidence ÷ (seberapa umum B secara keseluruhan) |

**Kunci Lift:** Lift > 1 = hubungan **nyata**. Lift = 1 = **kebetulan**. Kita mengurutkan pakai **Lift**, bukan Confidence, karena Confidence menipu untuk item yang memang umum.

---

## 🗂️ 2. Struktur Phase 3 — dua notebook

Phase 3 punya **dua notebook** yang memakai **input hand-off yang SAMA** (`data_phase3_rules.parquet`, 7 atribut kategorikal dari Phase 1), tapi **tujuan berlawanan**:

| | `PA_association_rule_mining` | `PA_anomaly_pattern_mining_and_validation` |
|---|---|---|
| Peran | Deliverable **inti** Phase 3 | Ekstensi → kontribusi ke Phase 4 |
| `min_support` | **tinggi (0,01)** → pola **umum** | **rendah (0,0002)** → pola **langka** |
| Pakai `isFraud`? | ❌ tidak | ✅ hanya untuk validasi (setelah mining) |
| Output | tabel aturan + interpretasi bisnis | flag anomali per-transaksi |

**Logika besarnya:** *dari banyak → sedikit yang bermakna.* Kita peras 6,3 juta transaksi menjadi kombinasi sering → aturan → aturan kuat → beberapa "permata" yang layak dilaporkan.

---

# 🅰️ NOTEBOOK A — `PA_association_rule_mining`

**Tujuan:** menemukan pola co-occurrence non-trivial, label-free. **Deliverable:** `phase3_association_rules.csv`.

### Section 0–1 — Setup & Load
- **Input:** `data_phase3_rules.parquet` — 6.362.620 baris × **7 atribut** (type, amount_category, time_segment, drain_category, emptied_origin, orig_balance_consistency, dest_kind).
- **Proses:** memuat file, memverifikasi kontrak 7 kolom, memastikan `isFraud` tidak ada.
- **Output:** dataframe `df` kategorikal.
- **Insight:** ini **satu sumber kebenaran** dari Phase 1 — Pattern Analyst tidak merekayasa ulang fitur (menghindari *drift*).

### Section 2 — Discretization review
- **Proses:** tampilkan distribusi tiap atribut (bar chart) + **support tiap item tunggal** (baseline).
- **Output:** grafik + daftar frekuensi (mis. Working_Hours 70%, TRANSFER 8,4%, DEBIT 0,65%).
- **Insight:** frekuensi tunggal ini = **baseline** yang jadi pembanding Lift nanti. `amount_category` sengaja ~33/33/33 (qcut), jadi rule ber-confidence >> 33% ke suatu band nominal itu **benar-benar informatif**.

### Section 3 — One-hot (basket)
- **Proses:** ubah 7 kolom kategorikal jadi **basket 20 item** True/False (`pd.get_dummies`).
- **Output:** matriks 6,3 jt × 20.
- **Insight:** Apriori hanya mengerti "ada/tidak", jadi kategori harus jadi kolom biner dulu.

### Section 3b — Redundancy check (tambahan penting) 🆕
- **Proses:** ukur redundansi antar-atribut dengan **Cramér's V** (bukan Pearson — itu untuk numerik) + korelasi **phi** antar item one-hot.
- **Output nyata:**
  - `type ↔ dest_kind` = **1,00** → **redundan sempurna** (dest_kind = fungsi type).
  - `drain_category ↔ emptied_origin` = **0,82** → Emptied **selalu** Full_Drain.
- **Justifikasi:** kalau tidak dicek, aturan tautologis (mis. `PAYMENT → Dest_Merchant`) akan mendominasi top-rules dan **melanggar syarat "non-trivial"** rubrik.
- **Insight:** ini bukti kuantitatif untuk **menyaring** rule tautologis di Section 6.

### Section 4 — Apriori (frequent itemsets)
- **Input:** basket + `min_support = 0,01`.
- **Proses:** cari kombinasi item yang muncul di ≥ 1% transaksi.
- **Output:** **1.060 frequent itemsets** (19 tunggal, 127 pasangan, 368 tripel, 546 kuadruple).
- **Parameter & justifikasi:**
  | Parameter | Nilai | Kenapa |
  |---|---|---|
  | `min_support` | 0,01 | cukup rendah agar DEBIT (0,65%) terlihat, cukup tinggi buang noise |
  | `max_len` | 4 | rule maks 3+1 item → interpretable + pangkas ledakan kombinasi |
  | `low_memory` | True | mode default butuh **3,4 GiB** → `MemoryError`; ini proses satu-per-satu |

### Section 5 — Generate rules + S/C/L
- **Proses:** ubah frequent itemsets jadi aturan `IF→THEN`, hitung Support/Confidence/Lift.
- **Output:** **7.242 candidate rules**.
- **Insight:** banyak yang lift-nya tinggi TAPI tautologis (mis. `Emptied → Full_Drain`) — belum disaring.

### Section 6 — Filter (statistik + anti-tautologi) 🆕
- **Proses:** simpan rule yang: `lift>1,2`, `confidence>0,5`, `support>0,01`, consequent tunggal — **lalu buang tautologi** (pasangan `type↔dest_kind`, `drain↔emptied`, dan consequent `dest_kind`).
- **Output nyata:** dari 1.294 rule yang lolos statistik, **817 tautologis dibuang → 477 strong rules**.
- **Justifikasi:** menjamin deliverable **non-trivial** (syarat rubrik). Tanpa ini, top-rules penuh `→ Orig_Consistent` confidence 1,0 yang nol insight.

### Section 7 — Visualisasi
- **Output:** scatter Support×Confidence (ukuran=Lift) + bar top-10 by Lift.

### Section 8 — Deliverable (dokumentasi) 🆕
- **Proses:** dedup pasangan simetris + **diversity cap** (maks 2 rule per consequent) → 12 aturan.
- **Output:** `phase3_association_rules.csv` — **12 aturan beragam** (7 jenis consequent berbeda) + kolom interpretasi.
- **Contoh aturan hasil (terverifikasi):**
  - `TRANSFER + Full_Drain/Emptied + Working_Hours → High_Amount` (conf ~0,90, **lift 2,70**) ⚠️ paling relevan fraud.
  - `Emptied + Medium_Amount → CASH_OUT` (conf 0,75–0,83, lift ~2,1–2,4).
  - `Low_Amount + Orig_Consistent → PAYMENT` (conf ~0,91, lift 2,68).

### Section 9 — Findings
- **Insight utama:** perilaku **pengurasan saldo mengikat channel & nominal** — transfer/cash-out yang mengosongkan rekening hampir pasti bernilai besar. Pola ini **tak terlihat** dari menghitung tipe atau nominal secara terpisah, dan **persis populasi yang harus difokuskan Phase 4**.

---

# 🅱️ NOTEBOOK B — `PA_anomaly_pattern_mining_and_validation`

**Tujuan:** menemukan *signature* perilaku **langka**, memvalidasinya ke fraud, menghasilkan **flag** untuk Phase 4.

### Section 1–2 — Load hand-off + verifikasi
- **Input:** `data_phase3_rules.parquet` (7 atribut, **sama** dengan Notebook A) + `labels_validation.parquet` (`isFraud`, **dipisah**).
- **Proses:** memuat fitur & label secara terpisah; hanya memverifikasi kontrak (tidak rekayasa ulang).
- **Insight:** label **tidak** masuk proses mining — dipegang di variabel terpisah.

### Section 3–4 — Basket + Apriori (pola LANGKA)
- **Input:** basket 20 item + **`min_support = 0,0002`** (jauh lebih rendah dari Notebook A).
- **Proses:** menambang **kombinasi langka** — inilah kandidat anomali.
- **Output:** ~1.900+ frequent itemsets (termasuk yang jarang).
- **Justifikasi:** anomali **jarang**; support tinggi akan membuangnya. Support rendah **sengaja** menjaganya.

### Section 5–6 — Rules + VALIDASI post-mining
- **Proses:** untuk tiap pattern, hitung **`fraud_rate`** (% cocok yang benar fraud) & **`fraud_recall`** dengan menyandingkan ke `isFraud` — **pertama & satu-satunya kali** label dipakai.
- **Output:** tabel pola berperingkat menurut keselarasan fraud.
- **Insight:** pola seperti `Emptied + Orig_Consistent` (rekening dikosongkan dengan pembukuan yang rapi) **selaras hampir sempurna dengan fraud**.

### Section 7 — Hand-off flag
- **Proses:** pilih pattern dengan **`fraud_rate ≥ 0,5`** → gabung jadi `behavioural_anomaly_flag`.
- **Output:** `phase3_rule_based_anomaly_flags.parquet` — flag per-transaksi (**precision 70%, recall 98%**) → dipakai Phase 4.
- **Justifikasi ambang 0,5:** hanya pola yang **mayoritasnya fraud** yang jadi flag.

---

## ✅ 3. Kriteria "BENAR" (kapan Phase 3 dianggap sah)

1. **Diskretisasi bermakna** (bin punya arti, bukan asal potong). ✅
2. **Support/Confidence/Lift dihitung** untuk tiap aturan. ✅
3. **Aturan disaring** — hanya lift tinggi + **non-trivial** (tautologi dibuang lewat cek Cramér's V). ✅
4. **≥10 aturan terdokumentasi** dengan **interpretasi bisnis**. ✅ (12 aturan)
5. **Label-free** saat mining (aturan project). ✅
6. **Menjawab**: *"pola apa yang tak terlihat dari tabel biasa?"* ✅

---

## 🧠 4. Ringkasan alur berpikir (satu tarikan napas)

**Data mentah → diskretisasi (Phase 1) → basket 20 item → Apriori (cari yang sering) → aturan + S/C/L → saring yang kuat & non-trivial → dokumentasikan ≥10 aturan bermakna.** Notebook anomaly memakai jalur yang sama tapi dengan `min_support` rendah untuk mengejar **yang langka**, lalu memvalidasinya ke fraud.

## 🗣️ 5. Cara menjelaskan ke teman (5 kalimat)

> *"Kita seperti kasir yang menganalisis struk — tapi struk-nya transaksi keuangan. Tiap transaksi jadi 'keranjang berisi label' (jenis, nominal, jam, perilaku saldo). Algoritma **Apriori** mencari label yang **sering muncul bersama**, dinilai dengan **Support** (umum), **Confidence** (andal), **Lift** (kuat vs kebetulan). Kita saring yang kuat & tidak sepele, lalu dokumentasikan ≥10 aturan dengan artinya untuk bisnis — semua tanpa melihat label fraud. Notebook kedua memakai cara sama tapi mengejar pola langka, lalu mengeceknya ke fraud untuk membuat 'alarm' bagi Phase 4."*
