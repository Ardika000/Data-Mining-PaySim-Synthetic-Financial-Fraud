# -*- coding: utf-8 -*-
"""
Phase 5 — PaySim KDD Interactive Console (Python Dash)
Owner: Insight Communicator

Semua data adalah hasil ASLI Phase 1-4 (dihasilkan oleh ../phase5_prepare_data.ipynb).
Dashboard hanya membaca agregat KECIL ber-dimensi HARI, lalu MENJUMLAHKAN bin sesuai
rentang slider -> latency mendekati nol. Tak ada label fraud yang dipakai saat mining;
`isFraud` hanya untuk validasi.

Menjalankan:
    python ../phase5_prepare_data.ipynb   # (jalankan notebook dulu -> isi folder ../data)
    python app.py                          # lalu buka http://localhost:8050
"""
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table, no_update
from dash.dash_table.Format import Format, Group

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')

INDIGO, TEAL, AMBER, PURPLE, SLATE, RED, GREEN = '#4F46E5', '#0F766E', '#B45309', '#9333EA', '#64748B', '#DC2626', '#15803D'
PALETTE = [INDIGO, TEAL, AMBER, PURPLE, SLATE]

# ── Muat data (graceful jika notebook belum dijalankan) ──────────────────────
REQUIRED = ['kpis.json', 'type_distribution.csv', 'temporal.csv', 'daily.csv',
            'cluster_profiles.csv', 'cluster_detail.csv', 'cluster_by_day.csv',
            'cluster_type_by_day.csv', 'cluster_scatter.csv', 'anomaly_by_day_vote.csv',
            'anomaly_by_day_type.csv', 'anomaly_by_day_cluster.csv', 'data_explorer.csv',
            'rules.csv', 'rule_by_day.csv']
DATA_READY = all(os.path.exists(os.path.join(D, f)) for f in REQUIRED)

if DATA_READY:
    kpis      = json.load(open(os.path.join(D, 'kpis.json'), encoding='utf-8'))
    type_dist = pd.read_csv(os.path.join(D, 'type_distribution.csv'))
    temporal  = pd.read_csv(os.path.join(D, 'temporal.csv'))
    daily     = pd.read_csv(os.path.join(D, 'daily.csv'))
    profiles  = pd.read_csv(os.path.join(D, 'cluster_profiles.csv'))
    cdetail   = pd.read_csv(os.path.join(D, 'cluster_detail.csv'))
    cbd       = pd.read_csv(os.path.join(D, 'cluster_by_day.csv'))
    ctd       = pd.read_csv(os.path.join(D, 'cluster_type_by_day.csv'))
    scatter   = pd.read_csv(os.path.join(D, 'cluster_scatter.csv'))
    an_vote   = pd.read_csv(os.path.join(D, 'anomaly_by_day_vote.csv'))
    an_type   = pd.read_csv(os.path.join(D, 'anomaly_by_day_type.csv'))
    an_clu    = pd.read_csv(os.path.join(D, 'anomaly_by_day_cluster.csv'))
    explorer  = pd.read_csv(os.path.join(D, 'data_explorer.csv'))
    rules     = pd.read_csv(os.path.join(D, 'rules.csv'))
    rbd       = pd.read_csv(os.path.join(D, 'rule_by_day.csv'))
    NDAYS     = int(kpis.get('n_days', int(daily['day'].max())))
    CLUSTER_NAME = {int(r.cluster): r.name for r in profiles.itertuples()}
    try:
        sampling = json.load(open(os.path.join(D, 'sampling_report.json'), encoding='utf-8'))
    except Exception:
        sampling = None
    try:
        gains = pd.read_csv(os.path.join(D, 'gains_curve.csv'))
    except Exception:
        gains = None
else:
    kpis = {}; NDAYS = 31; CLUSTER_NAME = {}; sampling = None; gains = None

# ════════════════════════════════════════════════════════════════════════════
# TERJEMAHAN KE BAHASA AWAM (kunci permintaan: minim jargon)
# ════════════════════════════════════════════════════════════════════════════
TERM_MAP = {
    'type=CASH_OUT': 'Penarikan tunai', 'type=TRANSFER': 'Transfer', 'type=PAYMENT': 'Pembayaran',
    'type=CASH_IN': 'Setoran tunai', 'type=DEBIT': 'Debit',
    'amount_category=Low_Amount': 'Nominal kecil', 'amount_category=Medium_Amount': 'Nominal sedang',
    'amount_category=High_Amount': 'Nominal besar',
    'time_segment=Working_Hours': 'Jam kerja', 'time_segment=Evening': 'Sore/malam',
    'time_segment=Night': 'Tengah malam', 'time_segment=Morning': 'Pagi',
    'drain_category=Full_Drain': 'Menguras habis saldo', 'drain_category=Mid_Drain': 'Menguras sebagian saldo',
    'drain_category=Low_Drain': 'Saldo hampir utuh', 'drain_category=No_Drain': 'Saldo tak berkurang',
    'emptied_origin=Emptied': 'Rekening dikosongkan', 'emptied_origin=Not_Emptied': 'Rekening tak dikosongkan',
    'orig_balance_consistency=Orig_Consistent': 'Saldo pengirim wajar',
    'orig_balance_consistency=Orig_Inconsistent': 'Saldo pengirim janggal',
    'dest_kind=Dest_Merchant': 'Tujuan: merchant/toko', 'dest_kind=Dest_Customer': 'Tujuan: rekening pribadi',
}

def term(t):
    t = t.strip()
    return TERM_MAP.get(t, t.split('=')[-1].replace('_', ' ') if '=' in t else t)

def split_terms(s):
    return [term(p) for p in str(s).split(',') if p.strip()]

# Interpretasi bisnis plain-language per aturan (menggantikan jargon lift/baseline)
RULE_INSIGHT = {
    1: 'Kiriman nominal sedang ke rekening pribadi (saldo wajar) biasanya memindahkan sebagian besar isi rekening — perpindahan dana rutin antar orang.',
    2: 'Penarikan tunai di jam kerja yang menguras sebagian saldo hampir selalu punya catatan saldo yang konsisten — penarikan normal, bukan manipulasi.',
    3: 'Pembayaran sore/malam yang menguras sebagian saldo hampir selalu konsisten secara saldo — pola belanja wajar.',
    4: 'Pembayaran sore/malam dengan saldo wajar hampir selalu bernominal kecil — belanja konsumtif harian.',
    5: 'Pembayaran ke merchant/toko di sore/malam hampir pasti bernominal kecil — transaksi ritel biasa.',
    6: 'Transfer di jam kerja yang MENGOSONGKAN rekening hampir pasti bernominal besar. Inilah pola yang paling mirip pengurasan rekening — perlu diwaspadai.',
    7: 'Transfer jam kerja yang menguras HABIS saldo hampir pasti bernilai besar — kandidat kuat untuk pengawasan fraud.',
    8: 'Transaksi kecil dengan saldo wajar hampir selalu berupa pembayaran — inilah segmen paling aman.',
    9: 'Transaksi kecil bersaldo wajar di jam kerja hampir pasti pembayaran rutin.',
    10: 'Di sore/malam, nominal sedang yang mengosongkan rekening cenderung berupa penarikan tunai.',
    11: 'Nominal sedang yang mengosongkan rekening umumnya adalah penarikan tunai.',
    12: 'Penarikan tunai yang menguras sebagian saldo cenderung bernominal sedang.',
}
FRAUD_RULES = {6, 7}

def conf_phrase(c):  return f"± {round(c*100)} dari 100 kali polanya benar"
def lift_word(l):    return 'sangat kuat' if l >= 2.5 else ('kuat' if l >= 1.8 else 'cukup')
def lift_phrase(l):  return f"{l:.1f}× lebih sering dari kebetulan — {lift_word(l)}"
def supp_phrase(s):  return f"terjadi pada {s*100:.1f}% transaksi global"

# Cerita tiap cluster: (emoji+judul, Karakteristik, Insight, Implementasi)
CLUSTER_STORY = {
    0: ('🟦 Setoran ke rekening bersaldo besar',
        'Didominasi CASH_IN dari rekening yang saldo awalnya sangat tinggi (rata-rata jutaan). Ini arus dana MASUK — top-up / penerimaan.',
        'Fraud sangat rendah. Ini "transaksi normal" arus masuk; menjadi baseline pembanding bila muncul setoran janggal.',
        'Pantau bila tiba-tiba ada CASH_IN abnormal ke rekening yang biasanya pasif (bisa indikasi rekening penampung).'),
    1: ('🟩 Pembayaran ritel kecil',
        'Hampir seluruhnya PAYMENT bernominal kecil dari saldo yang juga kecil — belanja / konsumtif harian.',
        'Risiko fraud TERENDAH — segmen paling sehat dan paling banyak jumlahnya.',
        'Perlakukan mendekati whitelist: kurangi intensitas screening di sini untuk menekan false-positive dan menghemat effort tim.'),
    2: ('🟧 Perpindahan besar dari rekening bersaldo ~0',
        'CASH_OUT/TRANSFER bernominal besar, tetapi saldo awal ≈ 0 dan catatan saldonya sering tidak konsisten.',
        'Fraud rendah — mayoritas kejanggalan STRUKTURAL pencatatan (artefak data), bukan fraud sungguhan.',
        'Arahkan ke tim rekonsiliasi data, bukan tim fraud. Perbaiki kualitas pencatatan saldo di hulu.'),
    3: ('🟥 Penarikan yang MENGURAS saldo',
        'CASH_OUT/TRANSFER dengan rasio pengurasan TERTINGGI — sering menguras atau bahkan melebihi saldo. Titiknya paling menyebar dari pusat segmennya.',
        'Salah satu segmen paling berisiko — pola paling dekat dengan complete-account-drain fraud.',
        'Terapkan aturan real-time: rasio penguras tinggi + kecepatan transaksi → minta verifikasi tambahan (step-up auth) atau tahan sementara (hold).'),
    4: ('🟪 Transaksi kecil bersaldo wajar — penyusupan fraud',
        'Sekilas tampak normal (nominal kecil, saldo konsisten), TETAPI memuat fraud TERBANYAK lewat minoritas penarikan/transfer di dalamnya.',
        'Segmen paling berbahaya justru yang paling terlihat normal — tak terdeteksi bila hanya melihat tipe/nominal.',
        'Butuh screening berbasis PERILAKU (gabungan skor anomali + pelanggaran aturan), bukan sekadar filter nominal besar.'),
}

# ════════════════════════════════════════════════════════════════════════════
# HELPER AGREGASI PER-RENTANG-HARI (mesin latency-nol)
# ════════════════════════════════════════════════════════════════════════════
def clusters_for_days(d0, d1):
    d = cbd[(cbd.day >= d0) & (cbd.day <= d1)]
    g = d.groupby('cluster').agg(n=('n', 'sum'), fraud=('fraud', 'sum'),
                                 sum_amount=('sum_amount', 'sum'), sum_drain=('sum_drain', 'sum')).reset_index()
    g['mean_amount'] = (g.sum_amount / g.n).round(0)
    g['mean_drain']  = (g.sum_drain / g.n).round(2)
    g['fraud_rate']  = (g.fraud / g.n * 100).round(3)
    tot = g.n.sum()
    g['pct'] = (g.n / tot * 100).round(2) if tot else 0
    g['name'] = g.cluster.map(CLUSTER_NAME)
    return g

def cluster_comp(cl, d0, d1):
    d = ctd[(ctd.cluster == cl) & (ctd.day >= d0) & (ctd.day <= d1)]
    s = d.groupby('type').n.sum().sort_values(ascending=False)
    tot = s.sum()
    return ' · '.join(f'{t} {v/tot*100:.0f}%' for t, v in s.head(3).items()) if tot else '-'

def votes_for_days(d0, d1):
    d = an_vote[(an_vote.day >= d0) & (an_vote.day <= d1)]
    g = d.groupby('vote').agg(count=('count', 'sum'), fraud=('fraud', 'sum')).reset_index()
    tot = g['count'].sum()
    g['pct'] = (g['count'] / tot * 100).round(2) if tot else 0
    g['fraud_rate'] = (g.fraud / g['count'] * 100).round(3)
    return g

def anom_group(df, key, d0, d1):
    d = df[(df.day >= d0) & (df.day <= d1)]
    g = d.groupby(key).agg(n=('n', 'sum'), hc=('hc', 'sum'), fraud=('fraud', 'sum')).reset_index()
    g['anomaly_rate'] = (g.hc / g.n * 100).round(3)
    g['fraud_rate']   = (g.fraud / g.n * 100).round(3)
    return g

def rule_activity(rid, d0, d1):
    d = rbd[(rbd['rule_#'] == rid) & (rbd.day >= d0) & (rbd.day <= d1)]
    return int(d['n_ac'].sum()), int(d['n_total'].sum())

def range_total_tx(d0, d1):
    return int(daily[(daily.day >= d0) & (daily.day <= d1)]['volume'].sum())

# ════════════════════════════════════════════════════════════════════════════
# STYLING GRAFIK
# ════════════════════════════════════════════════════════════════════════════
def style_fig(fig, h=360):
    fig.update_layout(template='plotly_white', font=dict(family='Inter, sans-serif', size=12, color='#111322'),
                      colorway=PALETTE, height=h, margin=dict(t=30, r=14, b=10, l=10),
                      paper_bgcolor='white', plot_bgcolor='white', legend=dict(font=dict(size=11)))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor='#EEF0F4', zeroline=False)
    return fig

# ── Building blocks ──────────────────────────────────────────────────────────
def kpi(label, val, cap='', cls=''):
    return html.Div([html.Div(label, className='kpi-label'),
                     html.Div(val, className=f'kpi-val {cls}'),
                     html.Div(cap, className='kpi-cap') if cap else None],
                    className='kpi' + (' alert' if 'red' in cls else ''))

def seg(_id, options, value):
    return dcc.RadioItems(id=_id, value=value, className='seg-radio',
                          options=[{'label': o[1], 'value': o[0]} for o in options])

def card(children, cls=''):
    return html.Div(children, className='card ' + cls)

def callout(title, body, cls=''):
    return html.Div([html.Div(['💡 ' + title], className='callout-title'), html.P(body)],
                    className='callout ' + cls)

def day_slider(_id):
    return html.Div([
        html.Div([
            # html.Span('🗓️', className='ico'),
                  html.Span('Rentang waktu', className='sb-title')], className='sb-head'),
        dcc.RangeSlider(id=_id, min=1, max=NDAYS, value=[1, NDAYS], step=1,
                        marks={d: str(d) for d in range(1, NDAYS + 1, 3)},
                        tooltip={'placement': 'bottom', 'always_visible': True}),
    ], className='slider-band')

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR + TOPBAR
# ════════════════════════════════════════════════════════════════════════════
# NAV = [('ov', '▦', 'Ringkasan', 'Overview'),
#        ('cl', '◉', 'Segmentasi', 'Cluster'),
#        ('ru', '⇄', 'Pola Normal', 'Aturan'),
#        ('an', '◎', 'Deteksi Anomali', 'Anomaly'),
#        ('doc', '☰', 'Dokumentasi', 'Laporan')]
NAV = [('ov', 'Ringkasan'),
       ('cl', 'Segmentasi'),
       ('ru', 'Pola Normal'),
       ('an',  'Deteksi Anomali'),
       ('ins', 'Insight Bisnis'),
       ('doc',  'Dokumentasi')]

sidebar = html.Div([
    html.Div([html.Div('K', className='brand-logo'),
              html.Div([html.Div('KDD Console', className='brand-name'),
                        html.Div('PaySim · Phase 5', className='brand-sub')])], className='brand'),
    html.Div([html.Div([html.Span(lbl),],
                       id=f'nav-{code}', className='nav-item' + (' active' if code == 'ov' else ''), n_clicks=0)
              for code, lbl in NAV], className='nav'),
    html.Div([html.Div('G7', className='avatar-sm'),
              html.Div([html.Div('Group 7', style={'fontWeight': 600, 'color': '#374151'}),
                        html.Div('Owner · 7 OSI')])], className='sidebar-foot'),
], className='sidebar')

topbar = html.Div([
    html.H1('PaySim — Knowledge Discovery Console'),
    html.Div([html.Span('Latency interaksi', className='ctl-label', style={'marginRight': '6px'}),
              html.Span('siap', id='latency-badge', className='lat-badge',
                        title='Waktu respons interaksi terakhir (round-trip klik/slider), diukur langsung di browser.')],
             className='top-right'),
], className='topbar')

# ════════════════════════════════════════════════════════════════════════════
# "DI BALIK LAYAR" — jembatan cerita tiap menu ke fase sumbernya (untuk expo)
# key: (label fase, APA yang dilakukan, KENAPA, DAMPAK ke halaman ini)
# ════════════════════════════════════════════════════════════════════════════
BEHIND = {
    'ov': ('Fase 1 · Data Understanding & Preprocessing',
           'Kami membersihkan 6,3 juta transaksi, membuat fitur perilaku baru (rasio pengurasan saldo, kejanggalan saldo, kategori waktu & nominal), dan melakukan EDA menyeluruh.',
           'Data mentah penuh nilai ekstrem & saldo tak konsisten. Tanpa dibersihkan dan difitur-kan, pola tersembunyi tak akan terbaca oleh fase berikutnya.',
           'Distribusi jenis transaksi, denyut per jam, dan timeline harian di halaman ini adalah hasil langsung EDA Fase 1. Fitur turunannya menjadi bahan baku Fase 2–4.'),
    'cl': ('Fase 2 · Segmentation via Clustering',
           'Kami mengelompokkan seluruh 6,3 juta transaksi dengan K-Means menjadi 5 segmen — memilih K=5 lewat Elbow/Silhouette/Davies-Bouldin, lalu memvalidasinya dengan DBSCAN & Hierarchical.',
           'Agar risiko bisa dipetakan per pola perilaku (bukan per transaksi satuan), dengan segmen yang stabil dan bisa dibandingkan antar waktu.',
           'Peta segmen, tabel profil, dan kartu karakteristik di halaman ini adalah label cluster Fase 2. Slider hari hanya mengiris data berlabel itu — bukan meng-cluster ulang.'),
    'ru': ('Fase 3 · Association Rule Mining',
           'Kami menjalankan Apriori atas 7 atribut kategorikal, menghasilkan 12 aturan "jika–maka", lalu menyaring aturan sepele dengan uji redundansi Cramér\'s V.',
           'Untuk memetakan "wajah normal" transaksi. Transaksi yang melanggar pola kuat inilah yang menjadi kandidat janggal untuk Fase 4.',
           'Kartu JIKA→MAKA di halaman ini adalah 12 aturan Fase 3. Angka kekuatannya dihitung dari seluruh data; slider hari hanya menampilkan seberapa aktif tiap pola.'),
    'an': ('Fase 4 · Anomaly & Outlier Detection',
           'Kami menandai transaksi menyimpang dengan 3 metode (IQR, Z-score, Isolation Forest) + voting, lalu memvalidasinya dengan label asli hanya di akhir.',
           'Untuk menyaring kandidat janggal tanpa memakai label saat mining, lalu memisahkan "error data / langka tapi sah / sinyal risiko".',
           'Grafik kesepakatan, janggal-vs-penipuan, dan tabel klik-untuk-jelaskan di halaman ini adalah hasil Fase 4 (dihitung pada sampel representatif 1,2 juta).'),
    'ins': ('Fase 5 · Sintesis Fase 1–4 + kurva recall Fase 4',
            'Kami merangkum seluruh temuan Fase 1–4 ke bahasa bisnis dan membangun simulator dari kurva recall (gains) skor anomali.',
            'Tujuan Fase 5: mengomunikasikan pengetahuan ke audiens non-teknis secara meyakinkan dan actionable — bukan sekadar akurasi model.',
            'Enam kartu temuan adalah kesimpulan lintas fase; simulator "fokus pengawasan" ditenagai kurva gains dari skor anomali Fase 4.'),
}

def behind_scenes(key):
    if key not in BEHIND:
        return None
    phase, apa, kenapa, dampak = BEHIND[key]
    def col(h, b):
        return html.Div([html.Div(h, className='bts-h'), html.Div(b, className='bts-b')], className='bts-col')
    return html.Div([
        html.Div([html.Span('🔍 Di balik layar — dari mana isi menu ini?', className='card-title'),
                  html.Span(phase, className='bts-phase')], className='bts-head'),
        html.Div([col('1 · APA YANG KAMI LAKUKAN', apa),
                  col('2 · KENAPA', kenapa),
                  col('3 · DAMPAK KE HALAMAN INI', dampak)], className='bts-grid'),
    ], className='card bts')

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — RINGKASAN (non-teknis, banyak visual)
# ════════════════════════════════════════════════════════════════════════════
def page_overview():
    kpi_row = html.Div([
        kpi('Total Transaksi', f"{kpis['total_tx']/1e6:.2f} jt", 'dataset PaySim (± 30 hari)'),
        kpi('Dugaan Penipuan', f"{kpis['fraud_rate']}%", f"{kpis['fraud_count']:,} kasus (1 dari ~775)", 'red'),
        kpi('Segmen Perilaku', f"{kpis['n_clusters']}", 'pola cara uang bergerak'),
        kpi('Transaksi Janggal', f"{kpis['high_conf_pct']}%", f"≈ {kpis['high_conf_count_est']:,} tx disaring"),
        kpi('Ketajaman Deteksi', f"{kpis['auc']}", f"10% paling janggal memuat {kpis['top_decile_recall']}% penipuan"),
    ], className='kpi-row')

    # flow = card([
    #     html.Div('Bagaimana pengetahuan ini ditemukan (alur KDD)', className='card-title'),
    #     html.Div([
    #         html.Div([html.Div('FASE 1', className='n'), html.Div('Rapikan data', className='t'),
    #                   html.Div('6,3 jt transaksi dibersihkan & diberi fitur perilaku.', className='d')], className='flow-step'),
    #         html.Div('→', className='flow-arrow'),
    #         html.Div([html.Div('FASE 2', className='n'), html.Div('Kelompokkan', className='t'),
    #                   html.Div('Dibagi jadi 5 segmen perilaku (clustering).', className='d')], className='flow-step'),
    #         html.Div('→', className='flow-arrow'),
    #         html.Div([html.Div('FASE 3', className='n'), html.Div('Pola normal', className='t'),
    #                   html.Div('12 aturan "jika–maka" yang lazim terjadi.', className='d')], className='flow-step'),
    #         html.Div('→', className='flow-arrow'),
    #         html.Div([html.Div('FASE 4', className='n'), html.Div('Saring janggal', className='t'),
    #                   html.Div('3 metode menandai transaksi menyimpang.', className='d')], className='flow-step'),
    #     ], className='flow', style={'marginTop': '10px'}),
    # ])

    charts = html.Div([
        card([html.Div([html.Div('Jenis transaksi', className='card-title'),
                        html.Div([html.Span('Tampilkan', className='ctl-label'),
                                  seg('ov-metric', [('count', 'Jumlah'), ('fraud_rate', 'Tingkat penipuan')], 'count')],
                                 style={'display': 'flex', 'alignItems': 'center'})], className='card-head'),
              dcc.Graph(id='ov-type', config={'displayModeBar': False}),
            #   html.Div(['Penarikan tunai & pembayaran mendominasi volume. ',
            #             html.B('Penipuan hanya muncul di TRANSFER & penarikan tunai'), ' — jenis lain praktis nol.'],
            #            className='caption')
                       ]),
        card([html.Div('Waktu Transaksi', className='card-title'),
              dcc.Graph(id='ov-temporal', figure=fig_temporal(), config={'displayModeBar': False}),
            #   html.Div('Volume naik-turun mengikuti jam aktif. Inilah "normal" yang jadi pembanding saat mencari yang janggal.',
            #            className='caption')
                       ]),
    ], className='grid-2')

    timeline = card([html.Div([html.Div('Aktivitas transaksi sepanjang periode', className='card-title'),
                               html.Div([html.Span('Garis penipuan', className='ctl-label'),
                                         seg('ov-daily-metric', [('count', 'Jumlah'), ('rate', 'Rasio (%)')], 'count')],
                                        style={'display': 'flex', 'alignItems': 'center'})], className='card-head'),
                     dcc.Graph(id='ov-daily', config={'displayModeBar': False})])

    # insight = callout('Temuan kunci — "janggal" belum tentu "penipuan"',
    #                   'Transaksi DEBIT paling sering ditandai janggal, tetapi NOL kasus penipuan — langka tapi sah. '
    #                   'Penipuan sungguhan justru sering menyamar jadi transaksi yang terlihat biasa. Itulah kenapa kita perlu '
    #                   'clustering + deteksi anomali, bukan sekadar aturan "nominal besar = mencurigakan".', 'warn')

    return html.Div([
        # html.H2('Ringkasan & Gambaran Besar', className='page-title'),
        #              html.P('Sebelum berburu transaksi janggal: seberapa besar datanya, jenis transaksinya apa, '
        #                     'dan seperti apa pola "normal"-nya. Halaman ini dibuat agar mudah dipahami tanpa latar teknis.',
        #                     className='page-sub'),
                     kpi_row, charts, timeline])

def fig_temporal():
    f = go.Figure()
    f.add_bar(x=temporal['hour'], y=temporal['volume'], name='Jumlah transaksi', marker_color='#C7CBE8')
    f.add_scatter(x=temporal['hour'], y=temporal['fraud_rate'], name='Tingkat penipuan (%)',
                  mode='lines+markers', line=dict(color=RED, width=3), yaxis='y2')
    f = style_fig(f, 300)
    f.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False), legend=dict(orientation='h', y=1.18),
                    xaxis_title='Jam (0–23)')
    return f

def fig_overview_daily(metric='count'):
    is_rate = (metric == 'rate')
    ycol   = 'fraud_rate' if is_rate else 'fraud'
    lname  = 'Tingkat penipuan/hari (%)' if is_rate else 'Jumlah penipuan/hari (kasus)'
    y2ttl  = 'Tingkat penipuan (%)' if is_rate else 'Jumlah penipuan (kasus)'
    f = go.Figure()
    f.add_bar(x=daily['day'], y=daily['volume'], name='Volume transaksi/hari', marker_color='#D8DBF2')
    f.add_scatter(x=daily['day'], y=daily[ycol], name=lname,
                  mode='lines+markers', line=dict(color=RED, width=2), marker=dict(size=5), yaxis='y2')
    f = style_fig(f, 260)
    f.update_layout(yaxis=dict(title='Volume transaksi'),
                    yaxis2=dict(title=y2ttl, overlaying='y', side='right', showgrid=False, rangemode='tozero'),
                    xaxis=dict(title='Hari', dtick=2), legend=dict(orientation='h', y=1.22))
    return f

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — SEGMENTASI (slider hari; karakteristik+insight+implementasi)
# ════════════════════════════════════════════════════════════════════════════
def page_segmentation():
    scat = card([
        html.Div([html.Div('Peta segmen (proyeksi 2D)', className='card-title'),
                  html.Div([html.Span('Warnai', className='ctl-label'),
                            seg('cl-colorby', [('cluster', 'Segmen'), ('type', 'Jenis')], 'cluster'),
                            html.Span('Fokus', className='ctl-label', style={'marginLeft': '12px'}),
                            dcc.Dropdown(id='cl-filter', value='all', clearable=False, style={'width': '150px'},
                                         options=[{'label': 'Semua', 'value': 'all'}] +
                                                 [{'label': f'Segmen {c}', 'value': str(c)} for c in sorted(CLUSTER_NAME)])],
                           style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'})], className='card-head'),
        dcc.Graph(id='cl-scatter', config={'displayModeBar': False}),
        # html.Div('Tiap titik = satu transaksi. Yang berdekatan berperilaku mirip. Terlihat 5 "pulau" perilaku yang berbeda.',
        #          className='caption')
                 ])
    rail = html.Div([
        card([html.Div('SEGMEN DITEMUKAN', className='kpi-label'),
              html.Div(str(kpis['n_clusters']), className='seg-count'),
              html.Div('✓ K-Means · Elbow + Silhouette', className='chip-ok', style={'marginTop': '8px'}),
              html.Div([html.Div(style={'flex': '1', 'background': c}) for c in PALETTE[:len(CLUSTER_NAME)]], className='bar-multi')]),
        html.Div(id='cl-note', className='callout'),
    ], className='rail')
    return html.Div([
                    # html.H2('Segmentasi Perilaku', className='page-title'),
                    #  html.P('Transaksi dikelompokkan berdasarkan CARA uang bergerak — bukan sekadar besar nominal. '
                    #         'Geser rentang hari untuk melihat komposisi & risiko tiap segmen berubah.', className='page-sub'),
                     day_slider('cl-days'),
                     html.Div([scat, rail], className='grid-scatter'),
                     card([html.Div('Ringkasan semua segmen (pada rentang terpilih)', className='card-title',
                                    style={'marginBottom': '10px'}),
                           html.Div(id='cl-table')]),
                     card([html.Div([html.Div('Karakteristik tiap segmen', className='card-title'),
                                     html.Div([html.Span('Pilih segmen', className='ctl-label'),
                                               dcc.Dropdown(id='cd-pick', value=0, clearable=False, style={'width': '260px'},
                                                            options=[{'label': f"Segmen {c} — {CLUSTER_NAME[c]}", 'value': c}
                                                                     for c in sorted(CLUSTER_NAME)])],
                                              style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'})],
                                    className='card-head'),
                           html.Div(id='cd-detail')])])

def cluster_table(g):
    show = g[['cluster', 'name', 'n', 'pct', 'mean_amount', 'mean_drain', 'fraud_rate']].copy()
    show['cluster'] = 'C' + show['cluster'].astype(str)
    show.columns = ['Segmen', 'Nama Bisnis', 'Jumlah', '% Data', 'Rata-Rata Nominal', 'Rata-Rata Penguras', 'Penipuan %']
    numc = ['Jumlah', '% Data', 'Rata-Rata Nominal', 'Rata-Rata Penguras', 'Penipuan %']
    grouped = {'Jumlah', 'Rata-Rata Nominal'}   # pemisah ribuan (mis. 2.500.000)
    def _col(c):
        col = {'name': c, 'id': c}
        if c in grouped:
            col.update({'type': 'numeric', 'format': Format(group=Group.yes).group_delimiter('.')})
        return col
    return dash_table.DataTable(
        data=show.round(2).to_dict('records'),
        columns=[_col(c) for c in show.columns],
        style_as_list_view=True,
        style_cell={'padding': '11px 10px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '13px', 'border': 'none'},
        style_header={'backgroundColor': '#F7F7FB', 'color': '#6B7280', 'fontWeight': '700',
                      'textTransform': 'uppercase', 'fontSize': '11px', 'letterSpacing': '.04em',
                      'borderBottom': '1px solid #E8E8F1'},
        style_data={'borderBottom': '1px solid #F0F0F6'},
        style_cell_conditional=[{'if': {'column_id': c}, 'fontFamily': 'JetBrains Mono, monospace', 'textAlign': 'right'} for c in numc],
        style_data_conditional=[{'if': {'filter_query': '{Penipuan %} > 0.2'}, 'backgroundColor': '#FDECEC'},
                                {'if': {'filter_query': '{Penipuan %} > 0.2', 'column_id': 'Penipuan %'},
                                 'color': '#B91C1C', 'fontWeight': '700'}])

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 3 — POLA NORMAL / ATURAN ASOSIASI (IF → MAKA, bahasa bisnis)
# ════════════════════════════════════════════════════════════════════════════
def page_rules():
    return html.Div([
        # html.H2('Pola Normal Transaksi ("Jika … maka …")', className='page-title'),
                    #  html.P('Seperti kasir yang menemukan "yang beli roti + selai biasanya beli mentega", di sini kita temukan '
                    #         'kebiasaan transaksi: JIKA sebuah transaksi punya ciri tertentu, MAKA biasanya berakhir bagaimana. '
                    #         'Ini memetakan "wajah normal" data — transaksi yang MELANGGAR pola kuat inilah kandidat janggal.',
                    #         className='page-sub'),
                     day_slider('ru-days'),
                     card([html.Div([html.Div('Saring kekuatan pola', className='card-title'),
                                     html.Div([html.Span('Hanya tampilkan pola minimal', className='ctl-label'),
                                               html.Div(dcc.Slider(id='ru-lift', min=1.0, max=3.0, step=0.1, value=1.0,
                                                                   marks={1: 'semua', 1.8: 'kuat', 2.5: 'sangat kuat'},
                                                                   tooltip={'placement': 'bottom', 'always_visible': False}),
                                                        style={'width': '340px'})],
                                              style={'display': 'flex', 'alignItems': 'center'})], className='card-head'),
                           html.P(['Node kuning = kondisi "JIKA", node hijau = akibat "MAKA". Garis makin tebal = pola makin kuat.'],
                                  style={'color': '#6B7280', 'margin': '0 0 6px', 'fontSize': '13px'}),
                           dcc.Graph(id='ru-network', config={'displayModeBar': False})]),
                     html.Div(id='ru-count', className='caption', style={'marginBottom': '14px'}),
                     html.Div(id='ru-cards')])

def fig_rule_network(min_lift):
    r = rules[rules['lift'] >= min_lift].copy()
    if len(r) == 0:
        return style_fig(go.Figure(), 300)
    ants = list(dict.fromkeys([' + '.join(split_terms(a)) for a in r['antecedent']]))
    cons = list(dict.fromkeys([term(c) for c in r['consequent']]))
    yl = {a: i * 1.4 for i, a in enumerate(ants)}
    yr = {c: i * 1.6 for i, c in enumerate(cons)}
    lmax = rules['lift'].max()
    f = go.Figure()
    for _, row in r.iterrows():
        a = ' + '.join(split_terms(row['antecedent'])); c = term(row['consequent'])
        w = 1 + 5 * row['lift'] / lmax
        col = RED if row['rule_#'] in FRAUD_RULES else INDIGO
        f.add_scatter(x=[0, 1], y=[yl[a], yr[c]], mode='lines',
                      line=dict(width=w, color=col), opacity=0.25 + 0.6 * row['lift'] / lmax,
                      hoverinfo='text', hovertext=f"JIKA {a}<br>MAKA {c}<br>kekuatan {row['lift']:.1f}×", showlegend=False)
    f.add_scatter(x=[0]*len(ants), y=list(yl.values()), mode='markers+text', text=ants, textposition='middle left',
                  marker=dict(size=12, color=AMBER), name='JIKA (kondisi)', hoverinfo='text')
    f.add_scatter(x=[1]*len(cons), y=list(yr.values()), mode='markers+text', text=cons, textposition='middle right',
                  marker=dict(size=13, color=TEAL), name='MAKA (akibat)', hoverinfo='text')
    f = style_fig(f, 440)
    f.update_layout(xaxis=dict(visible=False, range=[-1.1, 2.1]), yaxis=dict(visible=False),
                    legend=dict(orientation='h', y=1.08), margin=dict(l=6, r=6, t=40, b=6))
    return f

def rule_card(row, d0, d1):
    rid = int(row['rule_#'])
    is_fraud = rid in FRAUD_RULES
    n_ac, n_tot = rule_activity(rid, d0, d1)
    share = (n_ac / n_tot * 100) if n_tot else 0
    top = html.Div([html.Span(f'Aturan {rid}', className='rule-badge')] +
                   ([html.Span('dekat pola penipuan', className='rule-tag')] if is_fraud else []),
                   className='rule-top')
    ifthen = html.Div([
        html.Div([html.Span('JIKA', className='lbl-if')] +
                 [html.Span(t, className='chip a') for t in split_terms(row['antecedent'])], className='if-row'),
        html.Div('↓', className='arrow-down'),
        html.Div([html.Span('MAKA', className='lbl-then'),
                  html.Span(term(row['consequent']), className='chip c')], className='then-row'),
    ], className='ifthen')
    insight = html.Div([RULE_INSIGHT.get(rid, row.get('business_interpretation', ''))], className='rule-insight')
    pills = html.Div([
        html.Div([html.Div('Seberapa umum', className='k'), html.Div(f"{row['support']*100:.1f}%", className='val'),
                  html.Div(supp_phrase(row['support']), className='ph')], className='mpill'),
        html.Div([html.Div('Seberapa yakin', className='k'), html.Div(f"{row['confidence']*100:.0f}%", className='val'),
                  html.Div(conf_phrase(row['confidence']), className='ph')], className='mpill'),
        html.Div([html.Div('Sekuat apa polanya', className='k'), html.Div(f"{row['lift']:.1f}×", className='val'),
                  html.Div(lift_phrase(row['lift']), className='ph')], className='mpill'),
    ], className='metric-pills')
    act = html.Div([
        html.Div(f'Aktivitas pada rentang ini: {n_ac:,} transaksi cocok ({share:.1f}% dari volume rentang)',
                 style={'fontSize': '11.5px', 'color': '#6B7280', 'marginBottom': '4px'}),
        html.Div(html.Div(style={'width': f'{min(share*4,100):.0f}%', 'height': '6px',
                                 'background': (RED if is_fraud else INDIGO), 'borderRadius': '4px'}),
                 style={'background': '#EEF0F4', 'borderRadius': '4px', 'height': '6px'}),
    ], className='spark-wrap')
    return html.Div([top, ifthen, insight, pills, act], className='rule-card' + (' fraud' if is_fraud else ''))

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 4 — DETEKSI ANOMALI (slider; klik-untuk-jelaskan)
# ════════════════════════════════════════════════════════════════════════════
DISPLAY_COLS = ['type', 'segmen', 'day', 'amount', 'saldo_awal', 'drain_ratio', 'jam',
                'cluster', 'anomaly_vote', 'anomali', 'fraud']
COL_NAMES = {'type': 'Jenis', 'segmen': 'Segmen', 'day': 'Hari', 'amount': 'Nominal',
             'saldo_awal': 'Saldo Awal', 'drain_ratio': 'Rasio Kuras', 'jam': 'Jam',
             'cluster': 'Segmen#', 'anomaly_vote': 'Suara Janggal', 'anomali': 'Janggal', 'fraud': 'Penipuan'}
NUM_DISPLAY = ['day', 'amount', 'saldo_awal', 'drain_ratio', 'jam', 'cluster', 'anomaly_vote', 'anomali', 'fraud']

def explorer_table():
    return dash_table.DataTable(
        id='ex-dt',
        columns=[{'name': COL_NAMES[c], 'id': c, 'type': 'numeric' if c in NUM_DISPLAY else 'text'} for c in DISPLAY_COLS],
        data=[], filter_action='native', sort_action='native', page_action='native', page_size=10,
        cell_selectable=True, style_as_list_view=True,
        style_cell={'padding': '9px 10px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '12.5px', 'border': 'none'},
        style_header={'backgroundColor': '#F7F7FB', 'color': '#6B7280', 'fontWeight': '700',
                      'textTransform': 'uppercase', 'fontSize': '10.5px', 'letterSpacing': '.04em'},
        style_data={'borderBottom': '1px solid #F0F0F6', 'cursor': 'pointer'},
        style_cell_conditional=[{'if': {'column_id': c}, 'fontFamily': 'JetBrains Mono, monospace', 'textAlign': 'right'} for c in NUM_DISPLAY],
        style_data_conditional=[
            {'if': {'filter_query': '{fraud} = 1'}, 'backgroundColor': '#FDECEC'},
            {'if': {'filter_query': '{fraud} = 1', 'column_id': 'fraud'}, 'color': '#B91C1C', 'fontWeight': '700'},
            {'if': {'filter_query': '{anomaly_vote} = 3', 'column_id': 'anomaly_vote'}, 'color': INDIGO, 'fontWeight': '700'}])

def explain_placeholder():
    return html.Div([html.Div('👆 Klik satu baris di tabel', className='callout-title'),
                     html.P('Pilih transaksi mana pun untuk melihat KENAPA ia disebut janggal (atau tidak): '
                            'metode apa yang menandainya, pemicu utamanya, dan apakah ternyata penipuan.')])

def page_anomaly():
    hi_seg = int((profiles['fraud_rate'] > 0.1).sum())
    kpi_row = html.Div([
        kpi('Tingkat penipuan global', f"{kpis['fraud_rate']}%", 'acuan populasi', 'red'),
        kpi('Segmen berisiko', f"{hi_seg}", 'penipuan > 0,1%'),
        kpi('Metode deteksi', '3', 'IQR · Z-score · Isolation Forest', 'indigo'),
        kpi('Diperiksa (sampel)', f"{kpis.get('anomaly_sample_n', 0)/1e6:.1f} jt", 'representatif thd 6,3 jt'),
    ], className='kpi-row')
    chart = card([html.Div([html.Div('Makin banyak metode setuju, makin besar peluang penipuan', className='card-title'),
                            html.Div([html.Span('Lihat per', className='ctl-label'),
                                      seg('an-view', [('vote', 'Kesepakatan'), ('type', 'Jenis'), ('cluster', 'Segmen')], 'vote')],
                                     style={'display': 'flex', 'alignItems': 'center'})], className='card-head'),
                  dcc.Graph(id='an-main', config={'displayModeBar': False}),
                  html.Div(id='an-caption', className='caption')])
    rail = html.Div([html.Div(id='an-key', className='callout warn'), html.Div(id='an-votes', className='card')], className='rail')
    explorer_block = card([
        html.Div('Jelajah transaksi — klik untuk penjelasan', className='card-title', style={'marginBottom': '8px'}),
        html.Div([html.Span('Tampilkan', className='ctl-label'),
                  seg('ex-filter', [('all', 'Semua'), ('fraud', 'Penipuan'), ('anom', 'Janggal'), ('normal', 'Normal')], 'all'),
                  html.Span('Jenis', className='ctl-label', style={'marginLeft': '14px'}),
                  dcc.Dropdown(id='ex-type', value='all', clearable=False, style={'width': '160px'},
                               options=[{'label': 'Semua jenis', 'value': 'all'}] +
                                       [{'label': t, 'value': t} for t in sorted(explorer['type'].unique())])],
                 style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'margin': '2px 0 10px', 'flexWrap': 'wrap'}),
        html.Div(id='ex-count', className='caption', style={'margin': '0 0 10px'}),
        explorer_table(),
        html.Div(id='ex-explain', className='callout', children=explain_placeholder(), style={'marginTop': '14px'}),
    ])
    return html.Div([
        # html.H2('Deteksi Anomali', className='page-title'),
        #              html.P('Tiga metode menandai transaksi yang menyimpang jauh — TANPA melihat label penipuan. '
        #                     'Label baru dibuka di akhir hanya untuk mengukur seberapa baik tebakan kita. '
        #                     'Geser rentang hari; klik sebuah baris untuk penjelasan lengkap.', className='page-sub'),
                     day_slider('an-days'),
                     kpi_row,
                     html.Div([chart, rail], className='grid-anom'),
                     explorer_block])

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 5 — INSIGHT BISNIS (Knowledge Discovery Report; storytelling non-teknis)
# ════════════════════════════════════════════════════════════════════════════
# 6 temuan: (emoji, judul, DUGAAN umum, KENYATAAN data, AKSI, bukti angka)
DISCOVERIES = [
    ('🎭', 'Yang aneh belum tentu jahat',
     'Transaksi yang tidak biasa pasti penipuan.',
     'DEBIT paling sering ditandai janggal, tapi NOL penipuan. Sebaliknya, penipuan penarikan tunai malah menyamar jadi terlihat normal.',
     'Jangan pakai aturan "aneh = blokir". Pisahkan "langka tapi sah" dari "sinyal risiko".',
     'DEBIT: tingkat penipuan 0,000%. TRANSFER: 0,769%. CASH_OUT: 0,184% (menyamar, tingkat janggalnya rendah).'),
    ('🥷', 'Penjahat bersembunyi di keramaian',
     'Penipuan pasti bernominal besar & mencolok.',
     'Segmen dengan penipuan TERBANYAK justru berisi transaksi kecil bersaldo wajar — terlihat paling normal. Ini tak terlihat dari jenis/nominal; baru ketahuan setelah dikelompokkan per perilaku.',
     'Pantau berdasarkan PERILAKU (segmen), bukan hanya besar nominal.',
     'Segmen "penyusupan fraud" punya tingkat penipuan tertinggi (~0,40%) padahal isinya transaksi kecil yang tampak normal.'),
    ('📉', 'Lonjakan penipuan di hari sepi itu ilusi',
     'Persentase penipuan meledak di akhir periode → ada serangan.',
     'Jumlah penipuan stabil ±270/hari. Yang runtuh adalah volume transaksi normal, jadi persentasenya membesar — bukan wabah.',
     'Baca jumlah absolut + konteks volume; jangan panik pada lonjakan rasio.',
     'Hari 31: hanya 272 transaksi, kebetulan semua penipuan → rasio 100%. Hari 7: 272 penipuan dari 420.583 → 0,065%. Jumlah penipuan sama.'),
    ('💸', 'Menguras rekening menandai nominalnya',
     'Tak ada pola khusus pada transfer.',
     'Transfer yang MENGOSONGKAN rekening di jam kerja hampir pasti bernominal besar (±90%, 2,7× di atas biasa) — pola terdekat dengan pengurasan rekening.',
     'Jadikan "transfer + kuras habis + jam kerja" aturan prioritas real-time.',
     'Aturan 6 & 7: confidence ±90%, kekuatan (lift) 2,7× — pola paling dekat dengan complete-account-drain fraud.'),
    ('🤝', 'Kesepakatan melipatgandakan keyakinan',
     'Satu detektor sudah cukup.',
     'Saat 3 metode deteksi sepakat, peluang penipuan melonjak puluhan kali lipat. Satu detektor saja mudah keliru.',
     'Gunakan voting; eskalasi transaksi yang disepakati ≥2 metode.',
     'Tingkat penipuan: 0 metode ±0,05% → 3 metode ±8% (± 62× lipat) — ditemukan tanpa memakai label.'),
    ('🎯', 'Cukup periksa sebagian kecil',
     'Harus memeriksa semua transaksi untuk menangkap penipuan.',
     '10% transaksi paling mencurigakan sudah memuat mayoritas seluruh penipuan — tanpa pernah melihat label. Coba sendiri di simulator di atas.',
     'Fokuskan tim ke skor teratas → tangkap mayoritas penipuan dengan sebagian kecil usaha.',
     'ROC-AUC ±0,94; 10% paling anomali memuat ~76% seluruh penipuan.'),
]
RECS = [
    ('Triase 10% teratas', 'Prioritaskan pemeriksaan pada transaksi dengan skor kecurigaan tertinggi — memuat mayoritas penipuan dengan usaha minimal.'),
    ('Awasi perilaku, bukan nominal', 'Fokus ke segmen "penyusupan" yang terlihat normal, bukan sekadar menyaring nominal besar.'),
    ('Aturan pengurasan real-time', 'Verifikasi/tahan transfer & penarikan yang menguras habis rekening di jam kerja.'),
    ('Pisahkan janggal vs penipuan', 'Perlakukan "langka tapi sah" berbeda dari "sinyal risiko" agar false-positive turun & analis tak kebanjiran.'),
    ('Eskalasi berbasis kesepakatan', 'Naikkan prioritas otomatis bila ≥2 metode deteksi sepakat menandai satu transaksi.'),
]

def disc_card(emoji, title, myth, reality, action, evidence):
    return html.Div([
        html.Div([html.Span(emoji, className='disc-ico'), html.Span(title, className='disc-title')], className='disc-head'),
        html.Div([html.Span('DUGAAN UMUM', className='tag-myth'), html.Span(myth)], className='disc-myth'),
        html.Div([html.Span('KATA DATA', className='tag-real'), html.Span(reality)], className='disc-reality'),
        html.Div([html.B('→ Aksi: '), action], className='disc-take'),
        html.Details([html.Summary('Lihat bukti angka'), html.P(evidence)], className='disc-details'),
    ], className='disc-card')

def gains_lookup(pct):
    i = (gains['pct_inspected'] - pct).abs().idxmin()
    return float(gains.loc[i, 'pct_fraud_caught'])

def fig_gains(pct, caught):
    f = go.Figure()
    f.add_scatter(x=gains['pct_inspected'], y=gains['pct_fraud_caught'], mode='lines', name='Sistem kita',
                  line=dict(color=INDIGO, width=3), fill='tozeroy', fillcolor='rgba(79,70,229,0.08)')
    f.add_scatter(x=[0, 100], y=[0, 100], mode='lines', name='Kalau asal periksa (acak)',
                  line=dict(color=SLATE, width=1.5, dash='dot'))
    f.add_scatter(x=[pct], y=[caught], mode='markers', marker=dict(size=13, color=RED), name='Pilihanmu')
    f = style_fig(f, 300)
    f.update_layout(xaxis=dict(title='% transaksi diperiksa', range=[0, 100]),
                    yaxis=dict(title='% penipuan tertangkap', range=[0, 101]),
                    legend=dict(orientation='h', y=1.14))
    return f

def page_insight():
    tdr = kpis.get('top_decile_recall', 76.4)
    hero = html.Div([
        html.Div('Knowledge Discovery Report', className='eyebrow'),
        html.Div('Apa yang kami temukan yang tak terlihat dari data mentah?', className='q'),
        html.Div(['Dari data mentah kamu hanya melihat ', html.B('jenis, nominal, dan waktu'),
                  '. Kami menemukan bahwa penipuan bukan soal "nominal besar", melainkan ', html.B('pola perilaku'),
                  f' — dan pola itu bisa disaring hingga cukup memeriksa ~10% transaksi untuk menangkap ~{tdr:.0f}% penipuan, tanpa satu pun label.'],
                 className='a'),
    ], className='ins-hero')

    if gains is not None:
        sim = card([
            html.Div('🎯 Simulator — seberapa fokus pengawasan kita?', className='card-title'),
            html.P('Geser: "berapa persen transaksi paling mencurigakan yang mau kita periksa?" — lalu lihat berapa persen '
                   'penipuan yang tertangkap. Inilah bukti bahwa kita tak perlu memeriksa semuanya.',
                   style={'color': '#6B7280', 'margin': '2px 0 10px', 'fontSize': '13px'}),
            html.Div([html.Span('Periksa transaksi paling mencurigakan sebanyak', className='ctl-label'),
                      html.Div(dcc.Slider(id='sim-slider', min=1, max=100, step=1, value=10,
                                          marks={1: '1%', 10: '10%', 25: '25%', 50: '50%', 100: '100%'},
                                          tooltip={'placement': 'bottom', 'always_visible': True}),
                               style={'flex': '1'})],
                     style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '30px'}),
            html.Div([
                html.Div([html.Div(id='sim-caught', className='num'),
                          html.Div('penipuan tertangkap', className='lbl')], className='sim-big'),
                html.Div(dcc.Graph(id='sim-fig', config={'displayModeBar': False}), style={'flex': '1', 'minWidth': '0'}),
            ], className='sim-row'),
            html.Div(id='sim-sentence', className='sim-note'),
        ], 'sim-card')
    else:
        sim = callout('Cukup periksa sebagian kecil',
                      f'10% transaksi paling mencurigakan sudah memuat ~{tdr:.0f}% seluruh penipuan. '
                      '(Jalankan notebook phase5_prepare_data.ipynb untuk mengaktifkan simulator interaktif di sini.)', 'good')

    disc = html.Div([disc_card(*d) for d in DISCOVERIES], className='disc-grid')
    recs = card([html.Div('Rekomendasi aksi — apa yang harus dilakukan', className='card-title', style={'marginBottom': '10px'}),
                 html.Div([html.Div([html.Div(t, className='rec-t'), html.Div(dsc, className='rec-d')], className='rec-item')
                           for t, dsc in RECS])])
    return html.Div([
        html.H2('Insight Bisnis', className='page-title'),
        html.P('Terjemahan temuan ke bahasa bisnis: apa yang tak terlihat dari data mentah, kenapa penting, dan apa yang harus '
               'dilakukan. Dibuat untuk audiens non-teknis.', className='page-sub'),
        hero, sim,
        html.Div('6 temuan yang tak terlihat dari data mentah', className='gal-title', style={'fontSize': '13px', 'marginTop': '18px'}),
        disc, recs])

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 6 — DOKUMENTASI (teknis lengkap, untuk laporan dosen)
# ════════════════════════════════════════════════════════════════════════════
def _phase(badge, title, sections):
    parts = [html.Div([html.Span(badge, className='phase-badge'),
                       html.Span(title, style={'fontWeight': 800, 'fontSize': '17px', 'marginLeft': '10px'})],
                      style={'marginBottom': '4px'})]
    for label, items in sections:
        parts.append(html.Div(label, className='subh'))
        parts.append(html.Ul([html.Li(x) for x in items], className='doc-ul'))
    return html.Div(parts, className='card')

def _sampling_card():
    if not sampling:
        return None
    ks = sampling.get('ks', {})
    rows = [html.Tr([html.Td('Ukuran'), html.Td(f"{sampling['sample_n']:,}", className='mono'),
                     html.Td(f"{sampling['pop_n']:,}", className='mono'), html.Td(f"{sampling['sample_frac_pct']}% populasi")])]
    rows += [html.Tr([html.Td(f'Jarak distribusi (KS) · {k}'), html.Td(f"{v}", className='mono'),
                      html.Td('≈ 0 = identik', colSpan=2)]) for k, v in ks.items()]
    rows.append(html.Tr([html.Td('Tingkat penipuan'), html.Td(f"{sampling['fraud_rate_sample']}%", className='mono'),
                         html.Td(f"{sampling['fraud_rate_pop']}%", className='mono'), html.Td('sampel vs populasi')]))
    return html.Div([
        html.Div([html.Span('Bukti', className='phase-badge'),
                  html.Span('Sampling untuk dashboard tetap representatif', style={'fontWeight': 800, 'fontSize': '17px', 'marginLeft': '10px'})],
                 style={'marginBottom': '6px'}),
        html.P(sampling.get('verdict', ''), style={'color': '#4B5563', 'fontSize': '13px', 'margin': '0 0 8px'}),
        html.Table([html.Thead(html.Tr([html.Th('Metrik'), html.Th('Sampel'), html.Th('Populasi'), html.Th('Catatan')])),
                    html.Tbody(rows)], className='doc-table'),
    ], className='card')

# Plot output asli tiap fase (diekstrak dari notebook -> assets/plots/). (nama, keterangan, sumber notebook)
DOC_PLOTS = {
    'p1': [
        ('p1_type_distribution.png', 'Distribusi jenis transaksi + porsi fraud per tipe. Fraud hanya di TRANSFER & CASH_OUT; tipe lain 0.', 'DE1_EDA_and_Data_Quality.ipynb'),
        ('p1_numeric_dist.png', 'Distribusi fitur numerik setelah transformasi log1p — menormalkan skew ekstrem sebelum scaling.', 'DE1_EDA_and_Data_Quality.ipynb'),
        ('p1_correlation.png', 'Matriks korelasi Pearson antar fitur numerik mentah — memetakan hubungan & redundansi antar saldo/nominal.', 'DE1_EDA_and_Data_Quality.ipynb'),
        ('p1_temporal.png', 'Analisis temporal: volume transaksi per jam + pola fraud — menetapkan "denyut normal" harian.', 'DE1_EDA_and_Data_Quality.ipynb'),
    ],
    'p2': [
        ('p2_pca_variance.png', 'Explained variance per komponen PCA. Setelah perbaikan scaling, PC1 turun ke ~34% (tak lagi didominasi satu fitur).', 'clustering.ipynb'),
        ('p2_elbow_silhouette.png', 'Penentuan K optimal: Elbow (siku di K=5), Silhouette, dan Davies-Bouldin (minimum di K=5 = 1,145) → K=5.', 'clustering.ipynb'),
        ('p2_kdistance.png', 'K-distance plot (k=10) untuk memilih eps DBSCAN (persentil-85).', 'clustering.ipynb'),
        ('p2_dbscan_scatter.png', 'DBSCAN pada proyeksi 2D → memisahkan inti kepadatan dari noise/outlier (~8,7%).', 'clustering.ipynb'),
        ('p2_dendrogram.png', 'Dendrogram 3 linkage (sampel) → validasi struktur pengelompokan lewat paradigma koneksi.', 'clustering.ipynb'),
        ('p2_profiling.png', 'Profiling karakteristik tiap cluster (heatmap fitur) — dasar penamaan bisnis segmen.', 'clustering.ipynb'),
        ('p2_type_by_cluster.png', 'Komposisi jenis transaksi per cluster — memperlihatkan identitas tiap segmen.', 'clustering.ipynb'),
        ('p2_cluster_box.png', 'Sebaran nilai fitur per cluster (box plot) — menegaskan perbedaan perilaku antar segmen.', 'clustering.ipynb'),
    ],
    'p3': [
        ('p3_univariate.png', 'Distribusi tiap atribut kategorikal + support item tunggal (baseline pembanding Lift).', 'PA_association_rule_mining.ipynb'),
        ('p3_redundancy.png', "Cek redundansi antar-atribut (Cramér's V): type↔dest_kind = 1,00 (redundan sempurna) → dasar filter anti-tautologi.", 'PA_association_rule_mining.ipynb'),
        ('p3_rule_space.png', 'Ruang aturan: Support × Confidence (warna/ukuran = Lift) + 10 aturan lift tertinggi.', 'PA_association_rule_mining.ipynb'),
    ],
    'p4': [
        ('p4_summary.png', 'Ringkasan anomali: distribusi vote (0–3), sebaran skor Isolation Forest (ambang 1%), dan high-conf anomaly rate per tipe & per cluster.', 'DE_PA_anomaly_detection.ipynb'),
    ],
}
_PLOTDIR = os.path.join(HERE, 'assets', 'plots')

def doc_gallery(key):
    items = [it for it in DOC_PLOTS.get(key, []) if os.path.exists(os.path.join(_PLOTDIR, it[0]))]
    if not items:
        return None
    figs = [html.Div('Output & plot notebook', className='gal-title')]
    for fname, cap, src in items:
        figs.append(html.Div([
            html.Img(src=app.get_asset_url('plots/' + fname), alt=cap),
            html.Div([cap, html.Span('sumber: ' + src, className='src')], className='cap'),
        ], className='doc-fig'))
    return html.Div(figs)

def page_doc():
    v3 = None
    try:
        vv = votes_for_days(1, NDAYS)
        v3 = float(vv.loc[vv['vote'] == 3, 'fraud_rate'].iloc[0])
    except Exception:
        v3 = 7.98
    P1 = _phase('Fase 1', 'Data Understanding & Preprocessing', [
        ('Tujuan', ['Menghasilkan dataset bersih & siap tambang; mendokumentasikan tiap keputusan dengan justifikasi.']),
        ('Yang dikerjakan', [
            'EDA menyeluruh: kualitas data, distribusi tiap tipe, analisis saldo-nol, korelasi, entropy fitur, pola per jam.',
            'Rekayasa fitur: errorBalanceOrig/Dest (deviasi logika saldo), balance_drain_ratio, has_zero_orig_balance, '
            'time_segment, amount_category + 4 fitur perilaku (drain_category, emptied_origin, orig_balance_consistency, dest_kind).',
            'Dua cabang keluaran: matriks numerik ter-scaling (untuk clustering) & 7 atribut kategorikal (untuk association).',
            'Label isFraud disimpan terpisah — tidak pernah ikut proses mining.']),
        ('Metode & parameter', [
            'Scaling: log1p (amount & saldo) + signed-log (fitur error) → StandardScaler → clip [-5, 5].',
            'Diskretisasi: amount pakai qcut 3 tertil seimbang; waktu di-cut jadi 3 segmen harian.']),
        ('Justifikasi (kenapa begitu)', [
            'StandardScaler menggantikan RobustScaler: errorBalanceDest ~65% bernilai 0 → IQR≈0 → RobustScaler meledakkan '
            'variansnya (338.966; PCA PC1 = 99,82% — satu fitur mendominasi). Setelah diperbaiki: semua varians ≈ 1, PC1 turun ke ~34%.',
            'qcut memberi bin seimbang (~33% tiap kelas) → memaksimalkan keinformatifan & mencegah bias Apriori.']),
        ('Hasil & insight EDA', [
            'Penipuan hanya di TRANSFER & CASH_OUT; DEBIT/CASH_IN/PAYMENT praktis 0 → fokus pengawasan menyempit.',
            'Banyak errorBalance ≠ 0 (artefak simulasi) → sinyal fitur, bukan alasan membuang baris.',
            'Sampling 100k terbukti representatif terhadap populasi (uji KS ≈ 0,004 — distribusi hampir identik).']),
    ])
    P2 = _phase('Fase 2', 'Segmentation via Clustering', [
        ('Tujuan', ['Menemukan pengelompokan alami perilaku transaksi & memberi nama bisnis tiap segmen.']),
        ('Yang dikerjakan', [
            'K-Means pada SELURUH 6,3 jt baris → 5 segmen bernama.',
            'DBSCAN (PCA-5D, sampel 50k) untuk deteksi noise/outlier; Hierarchical (sampel 5k, 3 linkage) untuk validasi struktur.',
            'Profiling tiap cluster + ekspor label & jarak-ke-centroid untuk dipakai Phase 4.']),
        ('Metode & parameter', [
            'K optimal = 5 (Elbow patah di K=5; Davies-Bouldin minimum 1,145; Silhouette plateau 0,367).',
            'DBSCAN: eps dari k-distance (k=10) persentil-85, min_samples=10 → noise 8,7%.',
            'DBSCAN/Hierarchical disampel karena kompleksitas O(n²)/O(n³); K-Means (linear) dijalankan penuh.']),
        ('Justifikasi', [
            'Tiga algoritma = triangulasi 3 paradigma (centroid / densitas / koneksi); kesepakatan memperkuat validitas.',
            'K=5 dipilih atas dasar parsimoni + interpretabilitas, bukan sekadar argmax silhouette.']),
        ('Hasil & insight', [
            'Lima segmen bernama (lihat tab Segmentasi). Segmen "penguras saldo" & "penyusupan fraud" paling berisiko.',
            'Segmen paling berbahaya justru terlihat paling normal → alasan clustering + anomali dibutuhkan.']),
    ])
    P3 = _phase('Fase 3', 'Association Rule Mining (Pola Normal)', [
        ('Tujuan', ['Menemukan pola co-occurrence non-trivial antar atribut — 100% tanpa label.']),
        ('Yang dikerjakan', [
            'One-hot 7 atribut kategorikal → basket 20 item.',
            "Cek redundansi antar-atribut dengan Cramér's V (bukan Pearson, karena kategorikal).",
            'Apriori → frequent itemsets → aturan + Support/Confidence/Lift → filter statistik + anti-tautologi → 12 aturan.']),
        ('Metode & parameter', [
            'min_support=0,01; max_len=4; low_memory=True — wajib pada skala 6,3 jt × 20 item agar tidak MemoryError (3,4 GiB).',
            'Filter: lift>1,2; confidence>0,5; support>0,01; consequent tunggal; buang pasangan redundan.']),
        ('Justifikasi', [
            'Ranking pakai Lift, bukan Confidence — Confidence menipu untuk item yang memang umum.',
            "Anti-tautologi: type↔dest_kind redundan sempurna (Cramér's V = 1,00); emptied ⊂ drain (V = 0,82) — dibuang."]),
        ('Hasil & insight', [
            '1.060 frequent itemsets → 7.242 kandidat → 477 strong rules → 12 aturan terdokumentasi.',
            'Insight utama: perilaku pengurasan saldo MENGIKAT channel ke nominal — transfer/cash-out yang mengosongkan '
            'rekening hampir pasti bernilai besar (aturan 6 & 7). Pola inilah yang paling dekat dengan fraud.']),
    ])
    P4 = _phase('Fase 4', 'Anomaly & Outlier Detection', [
        ('Tujuan', ['Menemukan record menyimpang & mengklasifikasi: data error / rare-but-legitimate / risk signal.']),
        ('Yang dikerjakan', [
            'IQR + Z-score pada fitur MENTAH (agar interpretable) + Isolation Forest pada fitur ter-scaling.',
            'Ensemble voting 0–3; high-confidence = ≥2 metode setuju. Cross-reference dengan cluster outlier Phase 2.',
            'Pola domain (complete-drain, night+high). Validasi post-hoc dengan isFraud, lalu ekspor anomaly report.']),
        ('Metode & parameter', [
            'IQR multiplier 3,0; Z-score threshold 3σ; Isolation Forest: contamination 0,01, max_samples 256, 100 trees, seed 42.',
            'Cakupan sengaja berbeda: IQR ~43% (longgar), Z-score ~4,4%, Isolation Forest 1% (ketat, multivariat).']),
        ('Justifikasi', [
            'IQR 3,0 (bukan 1,5) karena variasi finansial besar → menekan false-positive.',
            'Isolation Forest dipilih (bukan LOF/One-Class SVM) karena skalabel untuk jutaan baris & multivariat.',
            'isFraud hanya disentuh di validasi akhir — bukan target mining.']),
        ('Hasil & insight', [
            f'Fraud rate naik monoton 0,05% → ± {v3:.1f}% seiring kesepakatan metode (lift ± 62× saat 3 metode setuju); '
            f'ROC-AUC {kpis.get("auc", 0.946)}; 10% paling anomali memuat ~{kpis.get("top_decile_recall", 79)}% seluruh penipuan.',
            'Insight kritis: anomali ≠ penipuan — DEBIT paling sering ditandai tapi 0 kasus penipuan (rare-but-legitimate). '
            'Fraud CASH_OUT justru menyamar (anomaly rate rendah) → butuh ensemble + cross-reference.']),
    ])
    P5 = _phase('Fase 5', 'Visualization & Knowledge Presentation', [
        ('Tujuan', ['Mengomunikasikan temuan ke audiens non-teknis lewat dashboard interaktif yang cepat.']),
        ('Yang dikerjakan', [
            'Notebook phase5_prepare_data.ipynb: pra-agregasi data 6,3 jt → berkas kecil ber-dimensi HARI (± 0,6 MB).',
            'Dashboard Dash 5 halaman: slider waktu di Segmentasi/Pola/Anomali, kartu JIKA→MAKA berbahasa bisnis, '
            'dan tabel jelajah dengan klik-untuk-jelaskan.']),
        ('Justifikasi (pipelining & sampling)', [
            'Pra-agregasi + penjumlahan bin per hari → latency mendekati nol saat slider digeser (tak mengolah 6,3 jt di sisi pengguna).',
            'Rasio bersifat sample-invariant → angka dari data penuh tetap representatif saat rentang digeser.',
            'Anomali & scatter disampel (1,2 jt & 8.000) demi kecepatan; representativeness dibuktikan dengan uji KS (lihat kartu Bukti).']),
        ('Hasil & insight', [
            'Temuan actionable: fokuskan pengawasan pada transfer/cash-out yang MENGURAS rekening — bukan sekadar nominal besar — '
            'dan pisahkan "anomali" dari "penipuan" agar analis tidak kebanjiran false-positive.']),
    ])
    blocks = []
    for blk, key in [(P1, 'p1'), (P2, 'p2'), (P3, 'p3'), (P4, 'p4'), (P5, None)]:
        blocks.append(blk)
        gal = doc_gallery(key) if key else None
        if gal:
            blocks.append(html.Div(gal, className='card'))
    sc = _sampling_card()
    if sc: blocks.append(sc)
    return html.Div(html.Div([
        html.Div('Laporan Akhir', className='eyebrow'),
        # html.H1('Apa yang kami temukan yang tak terlihat dari data mentah?', className='report-h1'),
        html.P('Penjabaran lengkap tiap fase: tujuan, langkah, metode & parameter, justifikasi, hasil & insight — '
               'sesuai keluaran notebook Phase 1–5.', className='report-lead'),
        html.Hr(className='hr'), *blocks,
        html.Div([html.Span('Intinya: tanpa pernah memakai label penipuan saat mining, '),
                  html.Span('10%', className='hl'), html.Span(' transaksi paling janggal memuat sekitar '),
                  html.Span(f"{kpis.get('top_decile_recall', 76)}%", className='hl'),
                  html.Span(' seluruh penipuan (ketajaman '), html.Span(f"{kpis.get('auc', 0.94)}", className='hl'),
                  html.Span('). Di sanalah pengawasan sebaiknya difokuskan.')], className='report-quote'),
    ], className='report'))

# ════════════════════════════════════════════════════════════════════════════
# LAYOUT + NAVIGASI
# ════════════════════════════════════════════════════════════════════════════
app = Dash(__name__, suppress_callback_exceptions=True, title='PaySim KDD Console')
server = app.server

def page_not_ready():
    return html.Div([html.H2('Data belum disiapkan', className='page-title'),
                     callout('Jalankan notebook dulu',
                             'Folder ../data masih kosong. Buka dan jalankan seluruh sel '
                             'notebooks/Phase5/phase5_prepare_data.ipynb (env DataMining) untuk menghasilkan berkas agregat, '
                             'lalu muat ulang halaman ini.', 'warn')], className='content')

if DATA_READY:
    app.layout = html.Div([dcc.Store(id='page', data='ov'), sidebar,
                           html.Div([topbar, html.Div(id='content', className='content')], className='main')], className='app')
else:
    app.layout = html.Div([sidebar, html.Div([topbar, page_not_ready()], className='main')], className='app')

if DATA_READY:
    @app.callback(Output('page', 'data'), [Input(f'nav-{c}', 'n_clicks') for c, *_ in NAV], prevent_initial_call=True)
    def navigate(*_):
        return (ctx.triggered_id or 'nav-ov').replace('nav-', '')

    @app.callback([Output(f'nav-{c}', 'className') for c, *_ in NAV], Input('page', 'data'))
    def set_active(page):
        return ['nav-item active' if c == page else 'nav-item' for c, *_ in NAV]

    @app.callback(Output('content', 'children'), Input('page', 'data'))
    def render(page):
        comp = {'ov': page_overview, 'cl': page_segmentation, 'ru': page_rules,
                'an': page_anomaly, 'ins': page_insight, 'doc': page_doc}.get(page, page_overview)()
        bs = behind_scenes(page)
        if bs is not None:
            try:
                comp.children.append(bs)
            except Exception:
                pass
        return comp

    # ── Insight Bisnis: simulator fokus pengawasan ──
    @app.callback(Output('sim-caught', 'children'), Output('sim-sentence', 'children'), Output('sim-fig', 'figure'),
                  Input('sim-slider', 'value'))
    def cb_sim(pct):
        if gains is None:
            return no_update, no_update, no_update
        caught = gains_lookup(pct)
        sentence = ['Dengan memeriksa hanya ', html.B(f'{pct}%'), ' transaksi paling mencurigakan, tim menangkap sekitar ',
                    html.B(f'{caught:.0f}%'), ' dari SELURUH penipuan — sisa ', html.B(f'{100-pct}%'),
                    ' transaksi tak perlu disentuh. Efisiensi inilah nilai utama proyek ini.']
        return f'{caught:.0f}%', sentence, fig_gains(pct, caught)

    # ── Overview ──
    @app.callback(Output('ov-type', 'figure'), Input('ov-metric', 'value'))
    def cb_ov(metric):
        d = type_dist.sort_values(metric, ascending=False)
        y = 'count' if metric == 'count' else 'fraud_rate'
        f = go.Figure(go.Bar(x=d['type'], y=d[y], marker_color=INDIGO,
                             text=[f'{v:,.0f}' if metric == 'count' else f'{v:.3f}%' for v in d[y]], textposition='outside'))
        f = style_fig(f, 300)
        f.update_yaxes(title='Jumlah' if metric == 'count' else 'Tingkat penipuan (%)')
        return f

    @app.callback(Output('ov-daily', 'figure'), Input('ov-daily-metric', 'value'))
    def cb_ov_daily(metric):
        return fig_overview_daily(metric)

    # ── Segmentation ──
    @app.callback(Output('cl-scatter', 'figure'), Input('cl-colorby', 'value'), Input('cl-filter', 'value'), Input('cl-days', 'value'))
    def cb_scatter(colorby, filt, days):
        d = scatter[(scatter.day >= days[0]) & (scatter.day <= days[1])].copy()
        if filt != 'all':
            d = d[d['cluster'] == int(filt)]
        d['Segmen'] = d['cluster'].map(lambda c: f"C{c} · {CLUSTER_NAME.get(c, c)}")
        f = px.scatter(d, x='pc1', y='pc2', color=('Segmen' if colorby == 'cluster' else 'type'),
                       color_discrete_sequence=PALETTE, opacity=0.6)
        f.update_traces(marker=dict(size=5))
        f = style_fig(f, 430)
        f.update_layout(legend=dict(title='', orientation='v', font=dict(size=10.5)), xaxis_title='', yaxis_title='')
        return f

    @app.callback(Output('cl-table', 'children'), Output('cl-note', 'children'), Input('cl-days', 'value'))
    def cb_cltable(days):
        g = clusters_for_days(days[0], days[1]).sort_values('cluster')
        top = g.sort_values('fraud_rate').iloc[-1]
        note = [html.Div('💡 Catatan analis', className='callout-title'),
                html.P([f"Pada hari {days[0]}–{days[1]}, segmen paling berisiko adalah ",
                        html.B(f"C{int(top.cluster)} ({top['name']})"),
                        f" dengan tingkat penipuan {top.fraud_rate:.3f}% dan rata-rata pengurasan {top.mean_drain:.1f}×. "
                        "Segmen inilah yang paling layak diprioritaskan untuk investigasi."])]
        return cluster_table(g), note

    @app.callback(Output('cd-detail', 'children'), Input('cd-pick', 'value'), Input('cl-days', 'value'))
    def cb_cldetail(c, days):
        g = clusters_for_days(days[0], days[1])
        row = g[g.cluster == c]
        if len(row) == 0:
            return html.P('Tidak ada data pada rentang ini.')
        row = row.iloc[0]
        title, karak, insight, impl = CLUSTER_STORY.get(int(c), (CLUSTER_NAME.get(c, f'Segmen {c}'), '', '', ''))
        comp = cluster_comp(c, days[0], days[1])
        def stat(lbl, val):
            return html.Div([html.Div(lbl, className='kpi-label'),
                             html.Div(val, className='kpi-val', style={'fontSize': '18px'})], className='kpi')
        stats = html.Div([stat('Jumlah', f"{int(row['n']):,}"), stat('% Data', f"{row['pct']:.1f}%"),
                          stat('Rata Nominal', f"{int(row['mean_amount']):,}"), stat('Rata Penguras', f"{row['mean_drain']:.1f}×"),
                          stat('Penipuan', f"{row['fraud_rate']:.3f}%")], className='kpi-row', style={'marginTop': '10px'})
        risk_cls = 'impl' if row['fraud_rate'] <= 0.1 else 'impl'
        return html.Div([
            html.H3(title, style={'margin': '4px 0 2px'}),
            html.Div(f"Komposisi jenis (rentang ini): {comp}", className='caption'),
            stats,
            html.Div([html.Div('KARAKTERISTIK', className='fh karak'), html.Div(karak, className='fb')], className='story-field'),
            html.Div([html.Div('INSIGHT', className='fh insight'), html.Div(insight, className='fb')], className='story-field'),
            html.Div([html.Div('IMPLEMENTASI', className='fh impl'), html.Div(impl, className='fb')], className='story-field'),
        ])

    # ── Rules ──
    @app.callback(Output('ru-network', 'figure'), Output('ru-cards', 'children'), Output('ru-count', 'children'),
                  Input('ru-lift', 'value'), Input('ru-days', 'value'))
    def cb_rules(min_lift, days):
        r = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False)
        cards = html.Div([rule_card(row, days[0], days[1]) for _, row in r.iterrows()], className='rule-grid')
        tot = range_total_tx(days[0], days[1])
        warn = ' ⚠ Rentang sangat kecil — angka aktivitas bisa tidak stabil.' if tot < 20000 else ''
        msg = [html.B(f'{len(r)} pola'), f' ditampilkan (dari {len(rules)}). Total {tot:,} transaksi pada hari {days[0]}–{days[1]}.',
               html.B(warn, style={'color': AMBER})]
        return fig_rule_network(min_lift), cards, msg

    # ── Anomaly ──
    @app.callback(Output('an-main', 'figure'), Output('an-caption', 'children'), Output('an-key', 'children'),
                  Output('an-votes', 'children'), Input('an-view', 'value'), Input('an-days', 'value'))
    def cb_anom(view, days):
        vv = votes_for_days(days[0], days[1]).sort_values('vote')
        # key finding + vote list (selalu dari data rentang)
        try:
            v3 = float(vv.loc[vv['vote'] == 3, 'fraud_rate'].iloc[0]); mult = v3 / max(kpis['fraud_rate'], 1e-9)
            key = [html.Div('💡 Temuan kunci', className='callout-title'),
                   html.P([f"Saat KETIGA metode setuju, tingkat penipuan melonjak jadi ", html.B(f"{v3:.1f}%"),
                           f" (± {mult:.0f}× lipat rata-rata) — ", html.B('ditemukan tanpa memakai label'), "."])]
        except Exception:
            key = [html.Div('💡 Temuan kunci', className='callout-title'), html.P('Data rentang tidak cukup.')]
        vlist = [html.Div('TERDAPAT "JANGGAL" (0–3 metode)', className='kpi-label', style={'marginBottom': '6px'})]
        for r in vv.itertuples():
            vlist.append(html.Div([html.Span(f"{int(r.vote)} metode"), html.Span(f"{r.pct:.1f}%", className='v')], className='mini-row'))

        if view == 'vote':
            f = go.Figure()
            f.add_bar(x=vv['vote'], y=vv['pct'], name='% transaksi', marker_color='#D8DBF2')
            f.add_scatter(x=vv['vote'], y=vv['fraud_rate'], name='Tingkat penipuan (%)', mode='lines+markers',
                          line=dict(color=RED, width=4), marker=dict(size=9), yaxis='y2')
            f = style_fig(f, 420)
            f.update_layout(yaxis=dict(title='% transaksi'),
                            yaxis2=dict(title='Penipuan (%)', overlaying='y', side='right', showgrid=False),
                            legend=dict(orientation='h', y=1.1), xaxis=dict(title='Jumlah metode yang setuju (0–3)', dtick=1))
            cap = ['Garis merah naik terus ke kanan: makin banyak metode sepakat sebuah transaksi janggal, makin besar peluang penipuan.']
        else:
            key_col = 'type' if view == 'type' else 'cluster'
            g = anom_group(an_type if view == 'type' else an_clu, key_col, days[0], days[1])
            xc = g['type'] if view == 'type' else ('C' + g['cluster'].astype(str))
            f = go.Figure()
            f.add_bar(x=xc, y=g['anomaly_rate'], name='Tingkat janggal (%)', marker_color=AMBER)
            f.add_bar(x=xc, y=g['fraud_rate'], name='Tingkat penipuan (%)', marker_color=RED)
            f = style_fig(f, 420)
            f.update_layout(barmode='group', yaxis_title='%', legend=dict(orientation='h', y=1.1))
            cap = ['Batang oranye tinggi tapi merah pendek = "sering janggal, jarang penipuan" (mis. DEBIT: langka tapi sah).']
        return f, cap, key, html.Div(vlist)

    @app.callback(Output('ex-dt', 'data'), Output('ex-count', 'children'),
                  Input('ex-filter', 'value'), Input('ex-type', 'value'), Input('an-days', 'value'))
    def cb_ex(filt, typ, days):
        d = explorer[(explorer['day'] >= days[0]) & (explorer['day'] <= days[1])]
        if filt == 'fraud':   d = d[d['fraud'] == 1]
        elif filt == 'anom':  d = d[d['anomali'] == 1]
        elif filt == 'normal': d = d[(d['anomali'] == 0) & (d['fraud'] == 0)]
        if typ != 'all':      d = d[d['type'] == typ]
        msg = [f"Menampilkan {len(d):,} transaksi (hari {days[0]}–{days[1]}). ",
               html.B('Klik satu baris'), ' untuk penjelasan; ketik di kotak filter kolom untuk mencari.']
        return d.to_dict('records'), msg

    @app.callback(Output('ex-explain', 'children'), Input('ex-dt', 'active_cell'), State('ex-dt', 'derived_virtual_data'))
    def cb_explain(active, vdata):
        if not active or not vdata or active['row'] >= len(vdata):
            return explain_placeholder()
        r = vdata[active['row']]
        methods = []
        if r.get('is_iqr'):  methods.append('IQR — nilainya jauh di luar rentang wajar (kuartil)')
        if r.get('is_z'):    methods.append('Z-score — lebih dari 3 simpangan baku dari rata-rata')
        if r.get('is_iso'):  methods.append('Isolation Forest — kombinasi ciri-cirinya langka sehingga mudah "diisolasi"')
        vote = int(r.get('anomaly_vote', 0))
        head = f"Transaksi {r['type']} · nominal {int(r['amount']):,} · hari {int(r['day'])}"
        if vote == 0:
            body = [html.P([html.B('Bukan anomali. '), 'Tidak ada metode yang menandai transaksi ini — wajar dari semua sisi.'])]
        else:
            body = [html.P([html.B(f'Ditandai janggal oleh {vote} dari 3 metode:')]),
                    html.Ul([html.Li(m) for m in methods], style={'margin': '2px 0 8px', 'paddingLeft': '20px'}),
                    html.P([html.B('Pemicu utama: '), r.get('top_reason', '-'), '.']),
                    html.P([html.B('Konteks segmen: '),
                            f"masuk Segmen {int(r['cluster'])} ({r['segmen']}), berjarak {r.get('dist', 0)} dari pusat segmennya "
                            + ('— cukup jauh, menyimpang dari perilaku normal segmennya.' if r.get('dist', 0) > 3
                               else '— relatif dekat pusat, jadi janggal karena nilainya ekstrem, bukan posisinya.')])]
        verdict = ('Ternyata PENIPUAN (dicek dari label — hanya untuk validasi).' if r.get('fraud')
                   else 'Ternyata BUKAN penipuan (validasi) — bukti bahwa "janggal" tak selalu berarti "penipuan".')
        return html.Div([html.Div('🔎 ' + head, className='callout-title')] + body +
                        [html.P(verdict, style={'marginTop': '6px', 'fontWeight': 600,
                                                'color': (RED if r.get('fraud') else TEAL)})])

if __name__ == '__main__':
    app.run(debug=False, port=8050)
