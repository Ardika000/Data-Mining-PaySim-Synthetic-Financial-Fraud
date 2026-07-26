# -*- coding: utf-8 -*-
"""
Phase 5 — PaySim KDD Interactive Console (Python Dash) · bilingual ID/EN
Owner: Insight Communicator

Semua data hasil ASLI Phase 1-4 (../phase5_prepare_data.ipynb). Bahasa bisa diganti
ID<->EN lewat toggle di navbar (variabel global LANG + helper t()).
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

# ════════════════════════════════════════════════════════════════════════════
# BAHASA (i18n) — variabel global + helper
# ════════════════════════════════════════════════════════════════════════════
LANG = 'id'   # 'id' | 'en'  (diubah oleh toggle di navbar)

def t(id_txt, en_txt):
    """Pilih string sesuai bahasa aktif (dipakai di dalam fungsi/callback saat render)."""
    return en_txt if LANG == 'en' else id_txt

def pick(pair):
    """Pilih dari tuple (id, en) untuk konstanta modul dwibahasa."""
    return pair[1] if LANG == 'en' else pair[0]

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
# TERJEMAHAN ISTILAH TEKNIS -> BAHASA AWAM (dwibahasa)
# code : (indonesia, english)
# ════════════════════════════════════════════════════════════════════════════
TERM_MAP = {
    'type=CASH_OUT': ('Penarikan tunai', 'Cash-out'), 'type=TRANSFER': ('Transfer', 'Transfer'),
    'type=PAYMENT': ('Pembayaran', 'Payment'), 'type=CASH_IN': ('Setoran tunai', 'Cash-in'),
    'type=DEBIT': ('Debit', 'Debit'),
    'amount_category=Low_Amount': ('Nominal kecil', 'Small amount'),
    'amount_category=Medium_Amount': ('Nominal sedang', 'Medium amount'),
    'amount_category=High_Amount': ('Nominal besar', 'Large amount'),
    'time_segment=Working_Hours': ('Jam kerja', 'Working hours'),
    'time_segment=Evening': ('Sore/malam', 'Evening'), 'time_segment=Night': ('Tengah malam', 'Night'),
    'time_segment=Morning': ('Pagi', 'Morning'),
    'drain_category=Full_Drain': ('Menguras habis saldo', 'Empties the balance'),
    'drain_category=Mid_Drain': ('Menguras sebagian saldo', 'Drains part of balance'),
    'drain_category=Low_Drain': ('Saldo hampir utuh', 'Balance mostly intact'),
    'drain_category=No_Drain': ('Saldo tak berkurang', 'No balance drop'),
    'emptied_origin=Emptied': ('Rekening dikosongkan', 'Account emptied'),
    'emptied_origin=Not_Emptied': ('Rekening tak dikosongkan', 'Account not emptied'),
    'orig_balance_consistency=Orig_Consistent': ('Saldo pengirim wajar', 'Sender balance consistent'),
    'orig_balance_consistency=Orig_Inconsistent': ('Saldo pengirim janggal', 'Sender balance inconsistent'),
    'dest_kind=Dest_Merchant': ('Tujuan: merchant/toko', 'To: merchant'),
    'dest_kind=Dest_Customer': ('Tujuan: rekening pribadi', 'To: personal account'),
}

def term(x):
    x = x.strip()
    if x in TERM_MAP:
        return pick(TERM_MAP[x])
    return x.split('=')[-1].replace('_', ' ') if '=' in x else x

def split_terms(s):
    return [term(p) for p in str(s).split(',') if p.strip()]

# Interpretasi bisnis per aturan — (id, en)
RULE_INSIGHT = {
    1: ('Kiriman nominal sedang ke rekening pribadi (saldo wajar) biasanya memindahkan sebagian besar isi rekening — perpindahan dana rutin antar orang.',
        'A medium-amount transfer to a personal account (consistent balance) usually moves most of the balance — routine person-to-person movement.'),
    2: ('Penarikan tunai di jam kerja yang menguras sebagian saldo hampir selalu punya catatan saldo yang konsisten — penarikan normal, bukan manipulasi.',
        'A working-hours cash-out that drains part of the balance almost always has consistent balance records — normal, not manipulation.'),
    3: ('Pembayaran sore/malam yang menguras sebagian saldo hampir selalu konsisten secara saldo — pola belanja wajar.',
        'An evening payment that drains part of the balance is almost always balance-consistent — normal spending.'),
    4: ('Pembayaran sore/malam dengan saldo wajar hampir selalu bernominal kecil — belanja konsumtif harian.',
        'An evening payment with a consistent balance is almost always a small amount — everyday spending.'),
    5: ('Pembayaran ke merchant/toko di sore/malam hampir pasti bernominal kecil — transaksi ritel biasa.',
        'An evening payment to a merchant is almost certainly a small amount — ordinary retail.'),
    6: ('Transfer di jam kerja yang MENGOSONGKAN rekening hampir pasti bernominal besar. Inilah pola yang paling mirip pengurasan rekening — perlu diwaspadai.',
        'A working-hours transfer that EMPTIES the account is almost certainly a large amount. This is the pattern closest to account draining — watch it.'),
    7: ('Transfer jam kerja yang menguras HABIS saldo hampir pasti bernilai besar — kandidat kuat untuk pengawasan fraud.',
        'A working-hours transfer that FULLY drains the balance is almost certainly large — a strong candidate for fraud monitoring.'),
    8: ('Transaksi kecil dengan saldo wajar hampir selalu berupa pembayaran — inilah segmen paling aman.',
        'A small transaction with a consistent balance is almost always a payment — the safest segment.'),
    9: ('Transaksi kecil bersaldo wajar di jam kerja hampir pasti pembayaran rutin.',
        'A small, balance-consistent transaction during working hours is almost certainly a routine payment.'),
    10: ('Di sore/malam, nominal sedang yang mengosongkan rekening cenderung berupa penarikan tunai.',
         'In the evening, a medium amount that empties the account tends to be a cash-out.'),
    11: ('Nominal sedang yang mengosongkan rekening umumnya adalah penarikan tunai.',
         'A medium amount that empties the account is generally a cash-out.'),
    12: ('Penarikan tunai yang menguras sebagian saldo cenderung bernominal sedang.',
         'A cash-out that drains part of the balance tends to be a medium amount.'),
}
FRAUD_RULES = {6, 7}

def conf_phrase(c):  return t(f"± {round(c*100)} dari 100 kali polanya benar", f"± {round(c*100)} out of 100 times the pattern holds")
def lift_word(l):    return t('sangat kuat', 'very strong') if l >= 2.5 else (t('kuat', 'strong') if l >= 1.8 else t('cukup', 'moderate'))
def lift_phrase(l):  return t(f"{l:.1f}× lebih sering dari kebetulan — {lift_word(l)}", f"{l:.1f}× more often than chance — {lift_word(l)}")
def supp_phrase(s):  return t(f"terjadi pada {s*100:.1f}% transaksi global", f"occurs in {s*100:.1f}% of all transactions")

# Cerita tiap cluster: c -> ((judul,karak,insight,impl)_id, (..)_en)
CLUSTER_STORY = {
    0: (('🟦 Setoran ke rekening bersaldo besar',
         'Didominasi CASH_IN dari rekening yang saldo awalnya sangat tinggi (rata-rata jutaan). Ini arus dana MASUK — top-up / penerimaan.',
         'Fraud sangat rendah. Ini "transaksi normal" arus masuk; menjadi baseline pembanding bila muncul setoran janggal.',
         'Pantau bila tiba-tiba ada CASH_IN abnormal ke rekening yang biasanya pasif (bisa indikasi rekening penampung).'),
        ('🟦 Deposits into high-balance accounts',
         'Dominated by CASH_IN from accounts whose starting balance is very high (millions on average). This is INCOMING money — top-up / receipts.',
         'Very low fraud. This is the "normal" inflow; it serves as the baseline against which odd deposits stand out.',
         'Watch for sudden abnormal CASH_IN into normally-passive accounts (a possible mule/holding account).')),
    1: (('🟩 Pembayaran ritel kecil',
         'Hampir seluruhnya PAYMENT bernominal kecil dari saldo yang juga kecil — belanja / konsumtif harian.',
         'Risiko fraud TERENDAH — segmen paling sehat dan paling banyak jumlahnya.',
         'Perlakukan mendekati whitelist: kurangi intensitas screening di sini untuk menekan false-positive dan menghemat effort tim.'),
        ('🟩 Small retail payments',
         'Almost entirely small PAYMENT from small balances — everyday consumer spending.',
         'LOWEST fraud risk — the healthiest and most numerous segment.',
         'Treat close to a whitelist: reduce screening intensity here to cut false positives and save team effort.')),
    2: (('🟧 Perpindahan besar dari rekening bersaldo ~0',
         'CASH_OUT/TRANSFER bernominal besar, tetapi saldo awal ≈ 0 dan catatan saldonya sering tidak konsisten.',
         'Fraud rendah — mayoritas kejanggalan STRUKTURAL pencatatan (artefak data), bukan fraud sungguhan.',
         'Arahkan ke tim rekonsiliasi data, bukan tim fraud. Perbaiki kualitas pencatatan saldo di hulu.'),
        ('🟧 Large moves from ~0-balance accounts',
         'Large CASH_OUT/TRANSFER, but starting balance ≈ 0 and balance records are often inconsistent.',
         'Low fraud — mostly STRUCTURAL recording quirks (data artefacts), not real fraud.',
         'Route to the data-reconciliation team, not the fraud team. Fix upstream balance-recording quality.')),
    3: (('🟥 Penarikan yang MENGURAS saldo',
         'CASH_OUT/TRANSFER dengan rasio pengurasan TERTINGGI — sering menguras atau bahkan melebihi saldo. Titiknya paling menyebar dari pusat segmennya.',
         'Salah satu segmen paling berisiko — pola paling dekat dengan complete-account-drain fraud.',
         'Terapkan aturan real-time: rasio penguras tinggi + kecepatan transaksi → minta verifikasi tambahan (step-up auth) atau tahan sementara (hold).'),
        ('🟥 Balance-DRAINING withdrawals',
         'CASH_OUT/TRANSFER with the HIGHEST drain ratio — often draining or exceeding the balance. Its points scatter farthest from the segment centre.',
         'One of the riskiest segments — the pattern closest to complete-account-drain fraud.',
         'Apply a real-time rule: high drain ratio + transaction velocity → step-up authentication or a temporary hold.')),
    4: (('🟪 Transaksi kecil bersaldo wajar — penyusupan fraud',
         'Sekilas tampak normal (nominal kecil, saldo konsisten), TETAPI memuat fraud TERBANYAK lewat minoritas penarikan/transfer di dalamnya.',
         'Segmen paling berbahaya justru yang paling terlihat normal — tak terdeteksi bila hanya melihat tipe/nominal.',
         'Butuh screening berbasis PERILAKU (gabungan skor anomali + pelanggaran aturan), bukan sekadar filter nominal besar.'),
        ('🟪 Small, normal-looking transactions — fraud infiltration',
         'Looks normal at a glance (small amounts, consistent balances), BUT holds the MOST fraud via a minority of cash-outs/transfers inside it.',
         'The most dangerous segment is the one that looks most normal — invisible if you only look at type/amount.',
         'Needs BEHAVIOUR-based screening (anomaly score + rule violations combined), not just a large-amount filter.')),
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
    return ' · '.join(f'{tp} {v/tot*100:.0f}%' for tp, v in s.head(3).items()) if tot else '-'

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
# STYLING GRAFIK + building blocks
# ════════════════════════════════════════════════════════════════════════════
def style_fig(fig, h=360):
    fig.update_layout(template='plotly_white', font=dict(family='Inter, sans-serif', size=12, color='#111322'),
                      colorway=PALETTE, height=h, margin=dict(t=30, r=14, b=10, l=10),
                      paper_bgcolor='white', plot_bgcolor='white', legend=dict(font=dict(size=11)))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor='#EEF0F4', zeroline=False)
    return fig

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
        html.Div([html.Span(t('Rentang waktu', 'Time range'), className='sb-title')], className='sb-head'),
        dcc.RangeSlider(id=_id, min=1, max=NDAYS, value=[1, NDAYS], step=1,
                        marks={d: str(d) for d in range(1, NDAYS + 1, 3)},
                        tooltip={'placement': 'bottom', 'always_visible': True}),
    ], className='slider-band')

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR + TOPBAR
# ════════════════════════════════════════════════════════════════════════════
NAV = [('ov', 'Ringkasan', 'Overview'),
       ('cl', 'Segmentasi', 'Segmentation'),
       ('ru', 'Pola Normal', 'Normal Patterns'),
       ('an', 'Deteksi Anomali', 'Anomaly Detection'),
       ('ins', 'Insight Bisnis', 'Business Insight'),
       ('doc', 'Dokumentasi', 'Documentation')]

def nav_label(code):
    for c, idl, enl in NAV:
        if c == code:
            return enl if LANG == 'en' else idl
    return code

sidebar = html.Div([
    html.Div([html.Div('K', className='brand-logo'),
              html.Div([html.Div('KDD Console', className='brand-name'),
                        html.Div('PaySim · Phase 5', className='brand-sub')])], className='brand'),
    html.Div([html.Div([html.Span(idl, id=f'navlbl-{code}')],
                       id=f'nav-{code}', className='nav-item' + (' active' if code == 'ov' else ''), n_clicks=0)
              for code, idl, enl in NAV], className='nav'),
    html.Div([html.Div('G7', className='avatar-sm'),
              html.Div([html.Div('Group 7', style={'fontWeight': 600, 'color': '#374151'}),
                        html.Div('Owner · 7 OSI')])], className='sidebar-foot'),
], className='sidebar')

topbar = html.Div([
    html.H1('PaySim — Knowledge Discovery Console'),
    html.Div([
        dcc.RadioItems(id='lang-toggle', value='id', className='seg-radio',
                       options=[{'label': 'ID', 'value': 'id'}, {'label': 'EN', 'value': 'en'}]),
        html.Span('Latency', id='lat-label', className='ctl-label', style={'margin': '0 6px 0 14px'}),
        html.Span('siap', id='latency-badge', className='lat-badge',
                  title='Round-trip time of the last interaction, measured in the browser.'),
    ], className='top-right'),
], className='topbar')

# ════════════════════════════════════════════════════════════════════════════
# "DI BALIK LAYAR" — jembatan cerita tiap menu ke fase sumbernya
# key -> ((phase,apa,kenapa,dampak)_id, (..)_en)
# ════════════════════════════════════════════════════════════════════════════
BEHIND = {
    'ov': (('Fase 1 · Data Understanding & Preprocessing',
            'Kami membersihkan 6,3 juta transaksi, membuat fitur perilaku baru (rasio pengurasan saldo, kejanggalan saldo, kategori waktu & nominal), dan melakukan EDA menyeluruh.',
            'Data mentah penuh nilai ekstrem & saldo tak konsisten. Tanpa dibersihkan dan difitur-kan, pola tersembunyi tak akan terbaca oleh fase berikutnya.',
            'Distribusi jenis transaksi, denyut per jam, dan timeline harian di halaman ini adalah hasil langsung EDA Fase 1. Fitur turunannya menjadi bahan baku Fase 2–4.'),
           ('Phase 1 · Data Understanding & Preprocessing',
            'We cleaned 6.3M transactions, engineered new behaviour features (balance-drain ratio, balance inconsistency, time & amount categories), and ran thorough EDA.',
            'Raw data is full of extreme values and inconsistent balances. Without cleaning and feature engineering, hidden patterns would be invisible to later phases.',
            'The transaction-type distribution, hourly rhythm, and daily timeline here are direct outputs of Phase 1 EDA. Its derived features feed Phases 2–4.')),
    'cl': (('Fase 2 · Segmentation via Clustering',
            'Kami mengelompokkan seluruh 6,3 juta transaksi dengan K-Means menjadi 5 segmen — memilih K=5 lewat Elbow/Silhouette/Davies-Bouldin, lalu memvalidasinya dengan DBSCAN & Hierarchical.',
            'Agar risiko bisa dipetakan per pola perilaku (bukan per transaksi satuan), dengan segmen yang stabil dan bisa dibandingkan antar waktu.',
            'Peta segmen, tabel profil, dan kartu karakteristik di halaman ini adalah label cluster Fase 2. Slider hari hanya mengiris data berlabel itu — bukan meng-cluster ulang.'),
           ('Phase 2 · Segmentation via Clustering',
            'We clustered all 6.3M transactions with K-Means into 5 segments — choosing K=5 via Elbow/Silhouette/Davies-Bouldin, then validating with DBSCAN & Hierarchical.',
            'So risk can be mapped per behaviour pattern (not per single transaction), with stable segments comparable across time.',
            'The segment map, profile table, and characteristic cards here are Phase 2 cluster labels. The day slider only slices that labelled data — it does not re-cluster.')),
    'ru': (('Fase 3 · Association Rule Mining',
            'Kami menjalankan Apriori atas 7 atribut kategorikal, menghasilkan 12 aturan "jika–maka", lalu menyaring aturan sepele dengan uji redundansi Cramér\'s V.',
            'Untuk memetakan "wajah normal" transaksi. Transaksi yang melanggar pola kuat inilah yang menjadi kandidat janggal untuk Fase 4.',
            'Kartu JIKA→MAKA di halaman ini adalah 12 aturan Fase 3. Angka kekuatannya dihitung dari seluruh data; slider hari hanya menampilkan seberapa aktif tiap pola.'),
           ('Phase 3 · Association Rule Mining',
            'We ran Apriori over 7 categorical attributes, produced 12 "if–then" rules, then filtered trivial ones with a Cramér\'s V redundancy check.',
            'To map the "normal face" of transactions. Transactions that break a strong rule become anomaly candidates for Phase 4.',
            'The IF→THEN cards here are the 12 Phase 3 rules. Their strength is computed over all data; the day slider only shows how active each pattern is.')),
    'an': (('Fase 4 · Anomaly & Outlier Detection',
            'Kami menandai transaksi menyimpang dengan 3 metode (IQR, Z-score, Isolation Forest) + voting, lalu memvalidasinya dengan label asli hanya di akhir.',
            'Untuk menyaring kandidat janggal tanpa memakai label saat mining, lalu memisahkan "error data / langka tapi sah / sinyal risiko".',
            'Grafik kesepakatan, janggal-vs-penipuan, dan tabel klik-untuk-jelaskan di halaman ini adalah hasil Fase 4 (dihitung pada sampel representatif 1,2 juta).'),
           ('Phase 4 · Anomaly & Outlier Detection',
            'We flagged deviating transactions with 3 methods (IQR, Z-score, Isolation Forest) + voting, validating with true labels only at the end.',
            'To triage odd candidates without using labels during mining, then separate "data error / rare-but-legit / risk signal".',
            'The agreement chart, anomaly-vs-fraud, and click-to-explain table here are Phase 4 outputs (computed on a representative 1.2M sample).')),
    'ins': (('Fase 5 · Sintesis Fase 1–4 + kurva recall Fase 4',
             'Kami merangkum seluruh temuan Fase 1–4 ke bahasa bisnis dan membangun simulator dari kurva recall (gains) skor anomali.',
             'Tujuan Fase 5: mengomunikasikan pengetahuan ke audiens non-teknis secara meyakinkan dan actionable — bukan sekadar akurasi model.',
             'Enam kartu temuan adalah kesimpulan lintas fase; simulator "fokus pengawasan" ditenagai kurva gains dari skor anomali Fase 4.'),
            ('Phase 5 · Synthesis of Phases 1–4 + Phase 4 recall curve',
             'We summarised all Phase 1–4 findings into business language and built a simulator from the anomaly-score recall (gains) curve.',
             'Phase 5 goal: communicate the knowledge to a non-technical audience convincingly and actionably — not merely model accuracy.',
             'The six finding cards are cross-phase conclusions; the "supervision focus" simulator is powered by the Phase 4 anomaly-score gains curve.')),
}

def behind_scenes(key):
    if key not in BEHIND:
        return None
    phase, apa, kenapa, dampak = pick(BEHIND[key])
    def col(h, b):
        return html.Div([html.Div(h, className='bts-h'), html.Div(b, className='bts-b')], className='bts-col')
    return html.Div([
        html.Div([html.Span(t('🔍 Di balik layar — dari mana isi menu ini?', '🔍 Behind the scenes — where does this menu come from?'), className='card-title'),
                  html.Span(phase, className='bts-phase')], className='bts-head'),
        html.Div([col(t('1 · APA YANG KAMI LAKUKAN', '1 · WHAT WE DID'), apa),
                  col(t('2 · KENAPA', '2 · WHY'), kenapa),
                  col(t('3 · DAMPAK KE HALAMAN INI', '3 · IMPACT ON THIS PAGE'), dampak)], className='bts-grid'),
    ], className='card bts')

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — RINGKASAN
# ════════════════════════════════════════════════════════════════════════════
def page_overview():
    kpi_row = html.Div([
        kpi(t('Total Transaksi', 'Total Transactions'), f"{kpis['total_tx']/1e6:.2f} jt" if LANG == 'id' else f"{kpis['total_tx']/1e6:.2f}M", t('dataset PaySim (± 30 hari)', 'PaySim dataset (~30 days)')),
        kpi(t('Dugaan Penipuan', 'Fraud Prevalence'), f"{kpis['fraud_rate']}%", t(f"{kpis['fraud_count']:,} kasus (1 dari ~775)", f"{kpis['fraud_count']:,} cases (1 in ~775)"), 'red'),
        kpi(t('Segmen Perilaku', 'Behaviour Segments'), f"{kpis['n_clusters']}", t('pola cara uang bergerak', 'patterns of money movement')),
        kpi(t('Transaksi Janggal', 'Anomalous Tx'), f"{kpis['high_conf_pct']}%", t(f"≈ {kpis['high_conf_count_est']:,} tx disaring", f"≈ {kpis['high_conf_count_est']:,} tx triaged")),
        kpi(t('Ketajaman Deteksi', 'Detection Sharpness'), f"{kpis['auc']}", t(f"10% paling janggal memuat {kpis['top_decile_recall']}% penipuan", f"top 10% holds {kpis['top_decile_recall']}% of fraud")),
    ], className='kpi-row')

    charts = html.Div([
        card([html.Div([html.Div(t('Jenis transaksi', 'Transaction types'), className='card-title'),
                        html.Div([html.Span(t('Tampilkan', 'Show'), className='ctl-label'),
                                  seg('ov-metric', [('count', t('Jumlah', 'Count')), ('fraud_rate', t('Tingkat penipuan', 'Fraud rate'))], 'count')],
                                 style={'display': 'flex', 'alignItems': 'center'})], className='card-head'),
              dcc.Graph(id='ov-type', config={'displayModeBar': False})]),
        card([html.Div(t('Waktu Transaksi', 'Transaction timing'), className='card-title'),
              dcc.Graph(id='ov-temporal', figure=fig_temporal(), config={'displayModeBar': False})]),
    ], className='grid-2')

    timeline = card([html.Div([html.Div(t('Aktivitas transaksi sepanjang periode', 'Transaction activity over the period'), className='card-title'),
                               html.Div([html.Span(t('Garis penipuan', 'Fraud line'), className='ctl-label'),
                                         seg('ov-daily-metric', [('count', t('Jumlah', 'Count')), ('rate', t('Rasio (%)', 'Rate (%)'))], 'count')],
                                        style={'display': 'flex', 'alignItems': 'center'})], className='card-head'),
                     dcc.Graph(id='ov-daily', config={'displayModeBar': False})])
    return html.Div([kpi_row, charts, timeline])

def fig_temporal():
    f = go.Figure()
    f.add_bar(x=temporal['hour'], y=temporal['volume'], name=t('Jumlah transaksi', 'Transaction count'), marker_color='#C7CBE8')
    f.add_scatter(x=temporal['hour'], y=temporal['fraud_rate'], name=t('Tingkat penipuan (%)', 'Fraud rate (%)'),
                  mode='lines+markers', line=dict(color=RED, width=3), yaxis='y2')
    f = style_fig(f, 300)
    f.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False), legend=dict(orientation='h', y=1.18),
                    xaxis_title=t('Jam (0–23)', 'Hour (0–23)'))
    return f

def fig_overview_daily(metric='count'):
    is_rate = (metric == 'rate')
    ycol = 'fraud_rate' if is_rate else 'fraud'
    lname = t('Tingkat penipuan/hari (%)', 'Fraud rate/day (%)') if is_rate else t('Jumlah penipuan/hari (kasus)', 'Fraud count/day (cases)')
    y2ttl = t('Tingkat penipuan (%)', 'Fraud rate (%)') if is_rate else t('Jumlah penipuan (kasus)', 'Fraud count (cases)')
    f = go.Figure()
    f.add_bar(x=daily['day'], y=daily['volume'], name=t('Volume transaksi/hari', 'Tx volume/day'), marker_color='#D8DBF2')
    f.add_scatter(x=daily['day'], y=daily[ycol], name=lname,
                  mode='lines+markers', line=dict(color=RED, width=2), marker=dict(size=5), yaxis='y2')
    f = style_fig(f, 260)
    f.update_layout(yaxis=dict(title=t('Volume transaksi', 'Tx volume')),
                    yaxis2=dict(title=y2ttl, overlaying='y', side='right', showgrid=False, rangemode='tozero'),
                    xaxis=dict(title=t('Hari', 'Day'), dtick=2), legend=dict(orientation='h', y=1.22))
    return f

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — SEGMENTASI
# ════════════════════════════════════════════════════════════════════════════
def page_segmentation():
    scat = card([
        html.Div([html.Div(t('Peta segmen (proyeksi 2D)', 'Segment map (2D projection)'), className='card-title'),
                  html.Div([html.Span(t('Warnai', 'Color by'), className='ctl-label'),
                            seg('cl-colorby', [('cluster', t('Segmen', 'Segment')), ('type', t('Jenis', 'Type'))], 'cluster'),
                            html.Span(t('Fokus', 'Focus'), className='ctl-label', style={'marginLeft': '12px'}),
                            dcc.Dropdown(id='cl-filter', value='all', clearable=False, style={'width': '160px'},
                                         options=[{'label': t('Semua', 'All'), 'value': 'all'}] +
                                                 [{'label': f"{t('Segmen','Segment')} {c}", 'value': str(c)} for c in sorted(CLUSTER_NAME)])],
                           style={'display': 'flex', 'alignItems': 'center', 'gap': '6px'})], className='card-head'),
        dcc.Graph(id='cl-scatter', config={'displayModeBar': False})])
    rail = html.Div([
        card([html.Div(t('SEGMEN DITEMUKAN', 'SEGMENTS FOUND'), className='kpi-label'),
              html.Div(str(kpis['n_clusters']), className='seg-count'),
              html.Div('✓ K-Means · Elbow + Silhouette', className='chip-ok', style={'marginTop': '8px'}),
              html.Div([html.Div(style={'flex': '1', 'background': c}) for c in PALETTE[:len(CLUSTER_NAME)]], className='bar-multi')]),
        html.Div(id='cl-note', className='callout'),
    ], className='rail')
    return html.Div([
        day_slider('cl-days'),
        html.Div([scat, rail], className='grid-scatter'),
        card([html.Div(t('Ringkasan semua segmen (pada rentang terpilih)', 'All segments summary (selected range)'), className='card-title',
                       style={'marginBottom': '10px'}),
              html.Div(id='cl-table')]),
        card([html.Div([html.Div(t('Karakteristik tiap segmen', 'Characteristics per segment'), className='card-title'),
                        html.Div([html.Span(t('Pilih segmen', 'Pick segment'), className='ctl-label'),
                                  dcc.Dropdown(id='cd-pick', value=0, clearable=False, style={'width': '280px'},
                                               options=[{'label': f"{t('Segmen','Segment')} {c} — {CLUSTER_NAME[c]}", 'value': c}
                                                        for c in sorted(CLUSTER_NAME)])],
                                 style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'})],
                       className='card-head'),
              html.Div(id='cd-detail')])])

def cluster_table(g):
    show = g[['cluster', 'name', 'n', 'pct', 'mean_amount', 'mean_drain', 'fraud_rate']].copy()
    show['cluster'] = 'C' + show['cluster'].astype(str)
    cols_id = ['Segmen', 'Nama Bisnis', 'Jumlah', '% Data', 'Rata-Rata Nominal', 'Rata-Rata Penguras', 'Penipuan %']
    cols_en = ['Segment', 'Business Name', 'Count', '% Data', 'Avg Amount', 'Avg Drain', 'Fraud %']
    cols = cols_en if LANG == 'en' else cols_id
    show.columns = cols
    c_count, c_amount, c_fraud = cols[2], cols[4], cols[6]
    numc = [cols[2], cols[3], cols[4], cols[5], cols[6]]
    grouped = {c_count, c_amount}
    def _col(c):
        col = {'name': c, 'id': c}
        if c in grouped:
            col.update({'type': 'numeric', 'format': Format(group=Group.yes).group_delimiter('.')})
        return col
    return dash_table.DataTable(
        data=show.round(2).to_dict('records'),
        columns=[_col(c) for c in cols],
        style_as_list_view=True,
        style_cell={'padding': '11px 10px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '13px', 'border': 'none'},
        style_header={'backgroundColor': '#F7F7FB', 'color': '#6B7280', 'fontWeight': '700',
                      'textTransform': 'uppercase', 'fontSize': '11px', 'letterSpacing': '.04em',
                      'borderBottom': '1px solid #E8E8F1'},
        style_data={'borderBottom': '1px solid #F0F0F6'},
        style_cell_conditional=[{'if': {'column_id': c}, 'fontFamily': 'JetBrains Mono, monospace', 'textAlign': 'right'} for c in numc],
        style_data_conditional=[{'if': {'filter_query': f'{{{c_fraud}}} > 0.2'}, 'backgroundColor': '#FDECEC'},
                                {'if': {'filter_query': f'{{{c_fraud}}} > 0.2', 'column_id': c_fraud},
                                 'color': '#B91C1C', 'fontWeight': '700'}])

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 3 — POLA NORMAL / ATURAN ASOSIASI
# ════════════════════════════════════════════════════════════════════════════
def page_rules():
    return html.Div([
        day_slider('ru-days'),
        card([html.Div([html.Div(t('Saring kekuatan pola', 'Filter pattern strength'), className='card-title'),
                        html.Div([html.Span(t('Kekuatan pola minimum', 'Minimum pattern strength'), className='ctl-label'),
                                  html.Div(dcc.Slider(id='ru-lift', min=1.0, max=3.0, step=0.1, value=1.0,
                                                      marks={1: t('semua', 'all'), 1.8: t('kuat', 'strong'), 2.5: t('sangat kuat', 'very strong')},
                                                      tooltip={'placement': 'bottom', 'always_visible': False}),
                                           style={'width': '340px'})],
                                 style={'display': 'flex', 'alignItems': 'center'})], className='card-head'),
              html.P([t('Node kuning = kondisi "JIKA", node hijau = akibat "MAKA". Garis makin tebal = pola makin kuat.',
                        'Amber node = "IF" condition, green node = "THEN" outcome. Thicker line = stronger pattern.')],
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
    IF, THEN = t('JIKA', 'IF'), t('MAKA', 'THEN')
    f = go.Figure()
    for _, row in r.iterrows():
        a = ' + '.join(split_terms(row['antecedent'])); c = term(row['consequent'])
        w = 1 + 5 * row['lift'] / lmax
        col = RED if row['rule_#'] in FRAUD_RULES else INDIGO
        f.add_scatter(x=[0, 1], y=[yl[a], yr[c]], mode='lines',
                      line=dict(width=w, color=col), opacity=0.25 + 0.6 * row['lift'] / lmax,
                      hoverinfo='text', hovertext=f"{IF} {a}<br>{THEN} {c}<br>{t('kekuatan','strength')} {row['lift']:.1f}×", showlegend=False)
    f.add_scatter(x=[0]*len(ants), y=list(yl.values()), mode='markers+text', text=ants, textposition='middle left',
                  marker=dict(size=12, color=AMBER), name=t('JIKA (kondisi)', 'IF (condition)'), hoverinfo='text')
    f.add_scatter(x=[1]*len(cons), y=list(yr.values()), mode='markers+text', text=cons, textposition='middle right',
                  marker=dict(size=13, color=TEAL), name=t('MAKA (akibat)', 'THEN (outcome)'), hoverinfo='text')
    f = style_fig(f, 440)
    f.update_layout(xaxis=dict(visible=False, range=[-1.1, 2.1]), yaxis=dict(visible=False),
                    legend=dict(orientation='h', y=1.08), margin=dict(l=6, r=6, t=40, b=6))
    return f

def rule_card(row, d0, d1):
    rid = int(row['rule_#'])
    is_fraud = rid in FRAUD_RULES
    n_ac, n_tot = rule_activity(rid, d0, d1)
    share = (n_ac / n_tot * 100) if n_tot else 0
    top = html.Div([html.Span(f"{t('Aturan','Rule')} {rid}", className='rule-badge')] +
                   ([html.Span(t('dekat pola penipuan', 'near fraud pattern'), className='rule-tag')] if is_fraud else []),
                   className='rule-top')
    ifthen = html.Div([
        html.Div([html.Span(t('JIKA', 'IF'), className='lbl-if')] +
                 [html.Span(x, className='chip a') for x in split_terms(row['antecedent'])], className='if-row'),
        html.Div('↓', className='arrow-down'),
        html.Div([html.Span(t('MAKA', 'THEN'), className='lbl-then'),
                  html.Span(term(row['consequent']), className='chip c')], className='then-row'),
    ], className='ifthen')
    insight = html.Div([pick(RULE_INSIGHT.get(rid, (row.get('business_interpretation', ''), row.get('business_interpretation', ''))))], className='rule-insight')
    pills = html.Div([
        html.Div([html.Div(t('Seberapa umum', 'How common'), className='k'), html.Div(f"{row['support']*100:.1f}%", className='val'),
                  html.Div(supp_phrase(row['support']), className='ph')], className='mpill'),
        html.Div([html.Div(t('Seberapa yakin', 'How reliable'), className='k'), html.Div(f"{row['confidence']*100:.0f}%", className='val'),
                  html.Div(conf_phrase(row['confidence']), className='ph')], className='mpill'),
        html.Div([html.Div(t('Sekuat apa polanya', 'How strong'), className='k'), html.Div(f"{row['lift']:.1f}×", className='val'),
                  html.Div(lift_phrase(row['lift']), className='ph')], className='mpill'),
    ], className='metric-pills')
    act = html.Div([
        html.Div(t(f'Aktivitas pada rentang ini: {n_ac:,} transaksi cocok ({share:.1f}% dari volume rentang)',
                   f'Activity in this range: {n_ac:,} matching transactions ({share:.1f}% of range volume)'),
                 style={'fontSize': '11.5px', 'color': '#6B7280', 'marginBottom': '4px'}),
        html.Div(html.Div(style={'width': f'{min(share*4,100):.0f}%', 'height': '6px',
                                 'background': (RED if is_fraud else INDIGO), 'borderRadius': '4px'}),
                 style={'background': '#EEF0F4', 'borderRadius': '4px', 'height': '6px'}),
    ], className='spark-wrap')
    return html.Div([top, ifthen, insight, pills, act], className='rule-card' + (' fraud' if is_fraud else ''))

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 4 — DETEKSI ANOMALI
# ════════════════════════════════════════════════════════════════════════════
DISPLAY_COLS = ['type', 'segmen', 'day', 'amount', 'saldo_awal', 'drain_ratio', 'jam',
                'cluster', 'anomaly_vote', 'anomali', 'fraud']
COL_NAMES = {
    'type': ('Jenis', 'Type'), 'segmen': ('Segmen', 'Segment'), 'day': ('Hari', 'Day'),
    'amount': ('Nominal', 'Amount'), 'saldo_awal': ('Saldo Awal', 'Start Balance'),
    'drain_ratio': ('Rasio Kuras', 'Drain Ratio'), 'jam': ('Jam', 'Hour'), 'cluster': ('Segmen#', 'Segment#'),
    'anomaly_vote': ('Suara Janggal', 'Anomaly Votes'), 'anomali': ('Janggal', 'Anomalous'), 'fraud': ('Penipuan', 'Fraud')}
NUM_DISPLAY = ['day', 'amount', 'saldo_awal', 'drain_ratio', 'jam', 'cluster', 'anomaly_vote', 'anomali', 'fraud']

def explorer_table():
    return dash_table.DataTable(
        id='ex-dt',
        columns=[{'name': pick(COL_NAMES[c]), 'id': c, 'type': 'numeric' if c in NUM_DISPLAY else 'text'} for c in DISPLAY_COLS],
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
    return html.Div([html.Div(t('👆 Klik satu baris di tabel', '👆 Click a row in the table'), className='callout-title'),
                     html.P(t('Pilih transaksi mana pun untuk melihat KENAPA ia disebut janggal (atau tidak): '
                              'metode apa yang menandainya, pemicu utamanya, dan apakah ternyata penipuan.',
                              'Pick any transaction to see WHY it is (or isn\'t) flagged: which method flagged it, '
                              'the main trigger, and whether it turned out to be fraud.'))])

def page_anomaly():
    hi_seg = int((profiles['fraud_rate'] > 0.1).sum())
    kpi_row = html.Div([
        kpi(t('Tingkat penipuan global', 'Global fraud rate'), f"{kpis['fraud_rate']}%", t('acuan populasi', 'population baseline'), 'red'),
        kpi(t('Segmen berisiko', 'Risky segments'), f"{hi_seg}", t('penipuan > 0,1%', 'fraud > 0.1%')),
        kpi(t('Metode deteksi', 'Detection methods'), '3', 'IQR · Z-score · Isolation Forest', 'indigo'),
        kpi(t('Diperiksa (sampel)', 'Checked (sample)'), f"{kpis.get('anomaly_sample_n', 0)/1e6:.1f} jt" if LANG == 'id' else f"{kpis.get('anomaly_sample_n', 0)/1e6:.1f}M", t('representatif thd 6,3 jt', 'representative of 6.3M')),
    ], className='kpi-row')
    chart = card([html.Div([html.Div(t('Makin banyak metode setuju, makin besar peluang penipuan', 'More methods agreeing → higher fraud probability'), className='card-title'),
                            html.Div([html.Span(t('Lihat per', 'View by'), className='ctl-label'),
                                      seg('an-view', [('vote', t('Kesepakatan', 'Agreement')), ('type', t('Jenis', 'Type')), ('cluster', t('Segmen', 'Segment'))], 'vote')],
                                     style={'display': 'flex', 'alignItems': 'center'})], className='card-head'),
                  dcc.Graph(id='an-main', config={'displayModeBar': False}),
                  html.Div(id='an-caption', className='caption')])
    rail = html.Div([html.Div(id='an-key', className='callout warn'), html.Div(id='an-votes', className='card')], className='rail')
    explorer_block = card([
        html.Div(t('Jelajah transaksi — klik untuk penjelasan', 'Explore transactions — click to explain'), className='card-title', style={'marginBottom': '8px'}),
        html.Div([html.Span(t('Tampilkan', 'Show'), className='ctl-label'),
                  seg('ex-filter', [('all', t('Semua', 'All')), ('fraud', t('Penipuan', 'Fraud')), ('anom', t('Janggal', 'Anomalous')), ('normal', 'Normal')], 'all'),
                  html.Span(t('Jenis', 'Type'), className='ctl-label', style={'marginLeft': '14px'}),
                  dcc.Dropdown(id='ex-type', value='all', clearable=False, style={'width': '170px'},
                               options=[{'label': t('Semua jenis', 'All types'), 'value': 'all'}] +
                                       [{'label': tp, 'value': tp} for tp in sorted(explorer['type'].unique())])],
                 style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'margin': '2px 0 10px', 'flexWrap': 'wrap'}),
        html.Div(id='ex-count', className='caption', style={'margin': '0 0 10px'}),
        explorer_table(),
        html.Div(id='ex-explain', className='callout', children=explain_placeholder(), style={'marginTop': '14px'}),
    ])
    return html.Div([
        day_slider('an-days'),
        kpi_row,
        html.Div([chart, rail], className='grid-anom'),
        explorer_block])

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 5 — INSIGHT BISNIS
# ════════════════════════════════════════════════════════════════════════════
# entry: (emoji, (judul,myth,reality,action,evidence)_id, (..)_en)
DISCOVERIES = [
    ('🎭',
     ('Yang aneh belum tentu jahat', 'Transaksi yang tidak biasa pasti penipuan.',
      'DEBIT paling sering ditandai janggal, tapi NOL penipuan. Sebaliknya, penipuan penarikan tunai malah menyamar jadi terlihat normal.',
      'Jangan pakai aturan "aneh = blokir". Pisahkan "langka tapi sah" dari "sinyal risiko".',
      'DEBIT: tingkat penipuan 0,000%. TRANSFER: 0,769%. CASH_OUT: 0,184% (menyamar, tingkat janggalnya rendah).'),
     ('Unusual isn\'t always bad', 'Anything unusual must be fraud.',
      'DEBIT is flagged anomalous most often, yet has ZERO fraud. Conversely, cash-out fraud disguises itself as normal-looking.',
      'Don\'t use "unusual = block". Separate "rare-but-legit" from "risk signal".',
      'DEBIT: fraud rate 0.000%. TRANSFER: 0.769%. CASH_OUT: 0.184% (disguised, low anomaly rate).')),
    ('🥷',
     ('Penjahat bersembunyi di keramaian', 'Penipuan pasti bernominal besar & mencolok.',
      'Segmen dengan penipuan TERBANYAK justru berisi transaksi kecil bersaldo wajar — terlihat paling normal. Ini tak terlihat dari jenis/nominal; baru ketahuan setelah dikelompokkan per perilaku.',
      'Pantau berdasarkan PERILAKU (segmen), bukan hanya besar nominal.',
      'Segmen "penyusupan fraud" punya tingkat penipuan tertinggi (~0,40%) padahal isinya transaksi kecil yang tampak normal.'),
     ('Criminals hide in the crowd', 'Fraud must be large and flashy.',
      'The segment with the MOST fraud actually holds small, balance-consistent transactions — the most normal-looking. Invisible from type/amount; only revealed after behaviour clustering.',
      'Monitor by BEHAVIOUR (segment), not just amount size.',
      'The "fraud infiltration" segment has the highest fraud rate (~0.40%) despite containing small, normal-looking transactions.')),
    ('📉',
     ('Lonjakan penipuan di hari sepi itu ilusi', 'Persentase penipuan meledak di akhir periode → ada serangan.',
      'Jumlah penipuan stabil ±270/hari. Yang runtuh adalah volume transaksi normal, jadi persentasenya membesar — bukan wabah.',
      'Baca jumlah absolut + konteks volume; jangan panik pada lonjakan rasio.',
      'Hari 31: hanya 272 transaksi, kebetulan semua penipuan → rasio 100%. Hari 7: 272 penipuan dari 420.583 → 0,065%. Jumlah penipuan sama.'),
     ('The quiet-day fraud spike is an illusion', 'Fraud % explodes at the end → an attack.',
      'Fraud count is stable at ±270/day. What collapses is normal transaction volume, so the percentage swells — not an outbreak.',
      'Read absolute counts + volume context; don\'t panic at ratio spikes.',
      'Day 31: only 272 transactions, all happen to be fraud → 100% rate. Day 7: 272 fraud out of 420,583 → 0.065%. Same fraud count.')),
    ('💸',
     ('Menguras rekening menandai nominalnya', 'Tak ada pola khusus pada transfer.',
      'Transfer yang MENGOSONGKAN rekening di jam kerja hampir pasti bernominal besar (±90%, 2,7× di atas biasa) — pola terdekat dengan pengurasan rekening.',
      'Jadikan "transfer + kuras habis + jam kerja" aturan prioritas real-time.',
      'Aturan 6 & 7: confidence ±90%, kekuatan (lift) 2,7× — pola paling dekat dengan complete-account-drain fraud.'),
     ('Draining reveals the amount', 'Transfers have no special pattern.',
      'A transfer that EMPTIES the account during working hours is almost certainly large (±90%, 2.7× above normal) — the pattern closest to account draining.',
      'Make "transfer + full drain + working hours" a real-time priority rule.',
      'Rules 6 & 7: confidence ±90%, strength (lift) 2.7× — the pattern closest to complete-account-drain fraud.')),
    ('🤝',
     ('Kesepakatan melipatgandakan keyakinan', 'Satu detektor sudah cukup.',
      'Saat 3 metode deteksi sepakat, peluang penipuan melonjak puluhan kali lipat. Satu detektor saja mudah keliru.',
      'Gunakan voting; eskalasi transaksi yang disepakati ≥2 metode.',
      'Tingkat penipuan: 0 metode ±0,05% → 3 metode ±8% (± 62× lipat) — ditemukan tanpa memakai label.'),
     ('Agreement multiplies confidence', 'One detector is enough.',
      'When 3 detection methods agree, fraud probability jumps dozens of times. A single detector is easily fooled.',
      'Use voting; escalate transactions agreed by ≥2 methods.',
      'Fraud rate: 0 methods ±0.05% → 3 methods ±8% (±62×) — found without using labels.')),
    ('🎯',
     ('Cukup periksa sebagian kecil', 'Harus memeriksa semua transaksi untuk menangkap penipuan.',
      '10% transaksi paling mencurigakan sudah memuat mayoritas seluruh penipuan — tanpa pernah melihat label. Coba sendiri di simulator di atas.',
      'Fokuskan tim ke skor teratas → tangkap mayoritas penipuan dengan sebagian kecil usaha.',
      'ROC-AUC ±0,94; 10% paling anomali memuat ~76% seluruh penipuan.'),
     ('Just check a small slice', 'You must inspect every transaction to catch fraud.',
      'The top 10% most suspicious transactions already hold the majority of all fraud — without ever seeing labels. Try the simulator above.',
      'Focus the team on the top scores → catch most fraud with a fraction of the effort.',
      'ROC-AUC ±0.94; the top 10% most anomalous hold ~76% of all fraud.')),
]
RECS = [
    (('Triase 10% teratas', 'Prioritaskan pemeriksaan pada transaksi dengan skor kecurigaan tertinggi — memuat mayoritas penipuan dengan usaha minimal.'),
     ('Triage the top 10%', 'Prioritise inspection of the highest-suspicion transactions — they hold most of the fraud with minimal effort.')),
    (('Awasi perilaku, bukan nominal', 'Fokus ke segmen "penyusupan" yang terlihat normal, bukan sekadar menyaring nominal besar.'),
     ('Watch behaviour, not amount', 'Focus on the normal-looking "infiltration" segment, not just filtering large amounts.')),
    (('Aturan pengurasan real-time', 'Verifikasi/tahan transfer & penarikan yang menguras habis rekening di jam kerja.'),
     ('Real-time drain rule', 'Verify/hold transfers & withdrawals that fully drain accounts during working hours.')),
    (('Pisahkan janggal vs penipuan', 'Perlakukan "langka tapi sah" berbeda dari "sinyal risiko" agar false-positive turun & analis tak kebanjiran.'),
     ('Separate anomalous vs fraud', 'Treat "rare-but-legit" differently from "risk signal" so false positives drop and analysts aren\'t flooded.')),
    (('Eskalasi berbasis kesepakatan', 'Naikkan prioritas otomatis bila ≥2 metode deteksi sepakat menandai satu transaksi.'),
     ('Agreement-based escalation', 'Auto-raise priority when ≥2 detection methods agree on flagging a transaction.')),
]

def disc_card(entry):
    emoji = entry[0]
    title, myth, reality, action, evidence = pick((entry[1], entry[2]))
    return html.Div([
        html.Div([html.Span(emoji, className='disc-ico'), html.Span(title, className='disc-title')], className='disc-head'),
        html.Div([html.Span(t('DUGAAN UMUM', 'COMMON BELIEF'), className='tag-myth'), html.Span(myth)], className='disc-myth'),
        html.Div([html.Span(t('KATA DATA', 'THE DATA SAYS'), className='tag-real'), html.Span(reality)], className='disc-reality'),
        html.Div([html.B(t('→ Aksi: ', '→ Action: ')), action], className='disc-take'),
        html.Details([html.Summary(t('Lihat bukti angka', 'See the numbers')), html.P(evidence)], className='disc-details'),
    ], className='disc-card')

def gains_lookup(pct):
    i = (gains['pct_inspected'] - pct).abs().idxmin()
    return float(gains.loc[i, 'pct_fraud_caught'])

def fig_gains(pct, caught):
    f = go.Figure()
    f.add_scatter(x=gains['pct_inspected'], y=gains['pct_fraud_caught'], mode='lines', name=t('Sistem kita', 'Our system'),
                  line=dict(color=INDIGO, width=3), fill='tozeroy', fillcolor='rgba(79,70,229,0.08)')
    f.add_scatter(x=[0, 100], y=[0, 100], mode='lines', name=t('Kalau asal periksa (acak)', 'Random checking'),
                  line=dict(color=SLATE, width=1.5, dash='dot'))
    f.add_scatter(x=[pct], y=[caught], mode='markers', marker=dict(size=13, color=RED), name=t('Pilihanmu', 'Your choice'))
    f = style_fig(f, 300)
    f.update_layout(xaxis=dict(title=t('% transaksi diperiksa', '% transactions inspected'), range=[0, 100]),
                    yaxis=dict(title=t('% penipuan tertangkap', '% fraud caught'), range=[0, 101]),
                    legend=dict(orientation='h', y=1.14))
    return f

def page_insight():
    tdr = kpis.get('top_decile_recall', 76.4)
    hero = html.Div([
        html.Div('Knowledge Discovery Report', className='eyebrow'),
        html.Div(t('Apa yang kami temukan yang tak terlihat dari data mentah?',
                   'What did we discover that wasn\'t obvious from the raw data?'), className='q'),
        html.Div(t([f'Dari data mentah kamu hanya melihat ', html.B('jenis, nominal, dan waktu'),
                    '. Kami menemukan bahwa penipuan bukan soal "nominal besar", melainkan ', html.B('pola perilaku'),
                    f' — dan pola itu bisa disaring hingga cukup memeriksa ~10% transaksi untuk menangkap ~{tdr:.0f}% penipuan, tanpa satu pun label.'],
                   [f'From raw data you only see ', html.B('type, amount, and time'),
                    '. We found that fraud isn\'t about "large amounts" but about ', html.B('behaviour patterns'),
                    f' — and those patterns can be triaged so that inspecting just ~10% of transactions catches ~{tdr:.0f}% of fraud, without any labels.']),
                 className='a'),
    ], className='ins-hero')

    if gains is not None:
        sim = card([
            html.Div(t('🎯 Simulator — seberapa fokus pengawasan kita?', '🎯 Simulator — how focused is our supervision?'), className='card-title'),
            html.P(t('Geser: "berapa persen transaksi paling mencurigakan yang mau kita periksa?" — lalu lihat berapa persen '
                     'penipuan yang tertangkap. Inilah bukti bahwa kita tak perlu memeriksa semuanya.',
                     'Slide: "what percent of the most suspicious transactions do we inspect?" — then see how much fraud is caught. '
                     'Proof that we don\'t need to check everything.'),
                   style={'color': '#6B7280', 'margin': '2px 0 10px', 'fontSize': '13px'}),
            html.Div([html.Span(t('Periksa transaksi paling mencurigakan sebanyak', 'Inspect the most suspicious transactions:'), className='ctl-label'),
                      html.Div(dcc.Slider(id='sim-slider', min=1, max=100, step=1, value=10,
                                          marks={1: '1%', 10: '10%', 25: '25%', 50: '50%', 100: '100%'},
                                          tooltip={'placement': 'bottom', 'always_visible': True}),
                               style={'flex': '1'})],
                     style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'marginBottom': '30px'}),
            html.Div([
                html.Div([html.Div(id='sim-caught', className='num'),
                          html.Div(t('penipuan tertangkap', 'fraud caught'), className='lbl')], className='sim-big'),
                html.Div(dcc.Graph(id='sim-fig', config={'displayModeBar': False}), style={'flex': '1', 'minWidth': '0'}),
            ], className='sim-row'),
            html.Div(id='sim-sentence', className='sim-note'),
        ], 'sim-card')
    else:
        sim = callout(t('Cukup periksa sebagian kecil', 'Just check a small slice'),
                      t(f'10% transaksi paling mencurigakan sudah memuat ~{tdr:.0f}% seluruh penipuan. '
                        '(Jalankan notebook phase5_prepare_data.ipynb untuk mengaktifkan simulator interaktif di sini.)',
                        f'The top 10% most suspicious transactions already hold ~{tdr:.0f}% of all fraud. '
                        '(Run phase5_prepare_data.ipynb to enable the interactive simulator here.)'), 'good')

    disc = html.Div([disc_card(d) for d in DISCOVERIES], className='disc-grid')
    recs = card([html.Div(t('Rekomendasi aksi — apa yang harus dilakukan', 'Action recommendations — what to do'), className='card-title', style={'marginBottom': '10px'}),
                 html.Div([html.Div([html.Div(pick(r)[0], className='rec-t'), html.Div(pick(r)[1], className='rec-d')], className='rec-item')
                           for r in RECS])])
    return html.Div([
        html.H2(t('Insight Bisnis', 'Business Insight'), className='page-title'),
        html.P(t('Terjemahan temuan ke bahasa bisnis: apa yang tak terlihat dari data mentah, kenapa penting, dan apa yang harus '
                 'dilakukan. Dibuat untuk audiens non-teknis.',
                 'Findings translated into business language: what wasn\'t obvious from raw data, why it matters, and what to do. '
                 'Built for a non-technical audience.'), className='page-sub'),
        hero, sim,
        html.Div(t('6 temuan yang tak terlihat dari data mentah', '6 findings invisible in the raw data'), className='gal-title', style={'fontSize': '13px', 'marginTop': '18px'}),
        disc, recs])

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 6 — DOKUMENTASI (teknis)
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
    rows = [html.Tr([html.Td(t('Ukuran', 'Size')), html.Td(f"{sampling['sample_n']:,}", className='mono'),
                     html.Td(f"{sampling['pop_n']:,}", className='mono'), html.Td(t(f"{sampling['sample_frac_pct']}% populasi", f"{sampling['sample_frac_pct']}% of population"))])]
    rows += [html.Tr([html.Td(t(f'Jarak distribusi (KS) · {k}', f'Distribution distance (KS) · {k}')), html.Td(f"{v}", className='mono'),
                      html.Td(t('≈ 0 = identik', '≈ 0 = identical'), colSpan=2)]) for k, v in ks.items()]
    rows.append(html.Tr([html.Td(t('Tingkat penipuan', 'Fraud rate')), html.Td(f"{sampling['fraud_rate_sample']}%", className='mono'),
                         html.Td(f"{sampling['fraud_rate_pop']}%", className='mono'), html.Td(t('sampel vs populasi', 'sample vs population'))]))
    return html.Div([
        html.Div([html.Span(t('Bukti', 'Evidence'), className='phase-badge'),
                  html.Span(t('Sampling untuk dashboard tetap representatif', 'Dashboard sampling remains representative'), style={'fontWeight': 800, 'fontSize': '17px', 'marginLeft': '10px'})],
                 style={'marginBottom': '6px'}),
        html.P(sampling.get('verdict', ''), style={'color': '#4B5563', 'fontSize': '13px', 'margin': '0 0 8px'}),
        html.Table([html.Thead(html.Tr([html.Th(t('Metrik', 'Metric')), html.Th(t('Sampel', 'Sample')), html.Th(t('Populasi', 'Population')), html.Th(t('Catatan', 'Note'))])),
                    html.Tbody(rows)], className='doc-table'),
    ], className='card')

# Plot output tiap fase: (nama, (cap_id, cap_en), sumber)
DOC_PLOTS = {
    'p1': [
        ('p1_type_distribution.png', ('Distribusi jenis transaksi + porsi fraud per tipe. Fraud hanya di TRANSFER & CASH_OUT; tipe lain 0.',
                                      'Transaction-type distribution + fraud share per type. Fraud only in TRANSFER & CASH_OUT; others zero.'), 'DE1_EDA_and_Data_Quality.ipynb'),
        ('p1_numeric_dist.png', ('Distribusi fitur numerik setelah transformasi log1p — menormalkan skew ekstrem sebelum scaling.',
                                 'Numeric feature distributions after log1p — normalising extreme skew before scaling.'), 'DE1_EDA_and_Data_Quality.ipynb'),
        ('p1_correlation.png', ('Matriks korelasi Pearson antar fitur numerik mentah — memetakan hubungan & redundansi antar saldo/nominal.',
                                'Pearson correlation matrix of raw numeric features — mapping relationships & redundancy among balances/amounts.'), 'DE1_EDA_and_Data_Quality.ipynb'),
        ('p1_temporal.png', ('Analisis temporal: volume transaksi per jam + pola fraud — menetapkan "denyut normal" harian.',
                             'Temporal analysis: hourly transaction volume + fraud pattern — establishing the daily "normal rhythm".'), 'DE1_EDA_and_Data_Quality.ipynb'),
    ],
    'p2': [
        ('p2_pca_variance.png', ('Explained variance per komponen PCA. Setelah perbaikan scaling, PC1 turun ke ~34% (tak lagi didominasi satu fitur).',
                                 'Explained variance per PCA component. After the scaling fix, PC1 drops to ~34% (no longer dominated by one feature).'), 'clustering.ipynb'),
        ('p2_elbow_silhouette.png', ('Penentuan K optimal: Elbow (siku di K=5), Silhouette, dan Davies-Bouldin (minimum di K=5 = 1,145) → K=5.',
                                     'Choosing optimal K: Elbow (knee at K=5), Silhouette, and Davies-Bouldin (minimum at K=5 = 1.145) → K=5.'), 'clustering.ipynb'),
        ('p2_kdistance.png', ('K-distance plot (k=10) untuk memilih eps DBSCAN (persentil-85).', 'K-distance plot (k=10) to choose DBSCAN eps (85th percentile).'), 'clustering.ipynb'),
        ('p2_dbscan_scatter.png', ('DBSCAN pada proyeksi 2D → memisahkan inti kepadatan dari noise/outlier (~8,7%).',
                                   'DBSCAN on the 2D projection → separating density cores from noise/outliers (~8.7%).'), 'clustering.ipynb'),
        ('p2_dendrogram.png', ('Dendrogram 3 linkage (sampel) → validasi struktur pengelompokan lewat paradigma koneksi.',
                               'Dendrogram, 3 linkages (sample) → validating cluster structure via the connectivity paradigm.'), 'clustering.ipynb'),
        ('p2_profiling.png', ('Profiling karakteristik tiap cluster (heatmap fitur) — dasar penamaan bisnis segmen.',
                              'Per-cluster characteristic profiling (feature heatmap) — the basis for business segment names.'), 'clustering.ipynb'),
        ('p2_type_by_cluster.png', ('Komposisi jenis transaksi per cluster — memperlihatkan identitas tiap segmen.',
                                    'Transaction-type composition per cluster — revealing each segment\'s identity.'), 'clustering.ipynb'),
        ('p2_cluster_box.png', ('Sebaran nilai fitur per cluster (box plot) — menegaskan perbedaan perilaku antar segmen.',
                                'Feature value spread per cluster (box plot) — confirming behaviour differences between segments.'), 'clustering.ipynb'),
    ],
    'p3': [
        ('p3_univariate.png', ('Distribusi tiap atribut kategorikal + support item tunggal (baseline pembanding Lift).',
                               'Distribution of each categorical attribute + single-item support (the baseline for Lift).'), 'PA_association_rule_mining.ipynb'),
        ('p3_redundancy.png', ("Cek redundansi antar-atribut (Cramér's V): type↔dest_kind = 1,00 (redundan sempurna) → dasar filter anti-tautologi.",
                               "Inter-attribute redundancy check (Cramér's V): type↔dest_kind = 1.00 (perfectly redundant) → basis for the anti-tautology filter."), 'PA_association_rule_mining.ipynb'),
        ('p3_rule_space.png', ('Ruang aturan: Support × Confidence (warna/ukuran = Lift) + 10 aturan lift tertinggi.',
                               'Rule space: Support × Confidence (colour/size = Lift) + the 10 highest-lift rules.'), 'PA_association_rule_mining.ipynb'),
    ],
    'p4': [
        ('p4_summary.png', ('Ringkasan anomali: distribusi vote (0–3), sebaran skor Isolation Forest (ambang 1%), dan high-conf anomaly rate per tipe & per cluster.',
                            'Anomaly summary: vote distribution (0–3), Isolation Forest score spread (1% threshold), and high-conf anomaly rate per type & per cluster.'), 'DE_PA_anomaly_detection.ipynb'),
    ],
}
_PLOTDIR = os.path.join(HERE, 'assets', 'plots')

def doc_gallery(key):
    items = [it for it in DOC_PLOTS.get(key, []) if os.path.exists(os.path.join(_PLOTDIR, it[0]))]
    if not items:
        return None
    figs = [html.Div(t('Output & plot notebook', 'Notebook outputs & plots'), className='gal-title')]
    for fname, cap_pair, src in items:
        cap = pick(cap_pair)
        figs.append(html.Div([
            html.Img(src=app.get_asset_url('plots/' + fname), alt=cap),
            html.Div([cap, html.Span(t('sumber: ', 'source: ') + src, className='src')], className='cap'),
        ], className='doc-fig'))
    return html.Div(figs)

def page_doc():
    try:
        vv = votes_for_days(1, NDAYS)
        v3 = float(vv.loc[vv['vote'] == 3, 'fraud_rate'].iloc[0])
    except Exception:
        v3 = 7.98
    P1 = _phase('Fase 1' if LANG == 'id' else 'Phase 1', 'Data Understanding & Preprocessing', [
        (t('Tujuan', 'Goal'), [t('Menghasilkan dataset bersih & siap tambang; mendokumentasikan tiap keputusan dengan justifikasi.',
                                  'Produce a clean, mining-ready dataset; document every decision with justification.')]),
        (t('Yang dikerjakan', 'What we did'), [
            t('EDA menyeluruh: kualitas data, distribusi tiap tipe, analisis saldo-nol, korelasi, entropy fitur, pola per jam.',
              'Thorough EDA: data quality, per-type distributions, zero-balance analysis, correlation, feature entropy, hourly patterns.'),
            t('Rekayasa fitur: errorBalanceOrig/Dest (deviasi logika saldo), balance_drain_ratio, has_zero_orig_balance, time_segment, amount_category + 4 fitur perilaku.',
              'Feature engineering: errorBalanceOrig/Dest (balance-logic deviation), balance_drain_ratio, has_zero_orig_balance, time_segment, amount_category + 4 behaviour features.'),
            t('Dua cabang keluaran: matriks numerik ter-scaling (untuk clustering) & 7 atribut kategorikal (untuk association).',
              'Two output branches: a scaled numeric matrix (for clustering) & 7 categorical attributes (for association).'),
            t('Label isFraud disimpan terpisah — tidak pernah ikut proses mining.', 'The isFraud label is kept separate — never used in mining.')]),
        (t('Metode & parameter', 'Method & parameters'), [
            t('Scaling: log1p (amount & saldo) + signed-log (fitur error) → StandardScaler → clip [-5, 5].',
              'Scaling: log1p (amount & balances) + signed-log (error features) → StandardScaler → clip [-5, 5].'),
            t('Diskretisasi: amount pakai qcut 3 tertil seimbang; waktu di-cut jadi 3 segmen harian.',
              'Discretisation: amount via balanced 3-tertile qcut; time cut into 3 daily segments.')]),
        (t('Justifikasi (kenapa begitu)', 'Justification (why)'), [
            t('StandardScaler menggantikan RobustScaler: errorBalanceDest ~65% bernilai 0 → IQR≈0 → RobustScaler meledakkan variansnya (338.966; PC1 = 99,82%). Setelah diperbaiki: semua varians ≈ 1, PC1 turun ke ~34%.',
              'StandardScaler replaced RobustScaler: errorBalanceDest is ~65% zeros → IQR≈0 → RobustScaler blows its variance up (338,966; PC1 = 99.82%). After the fix: all variances ≈ 1, PC1 drops to ~34%.'),
            t('qcut memberi bin seimbang (~33% tiap kelas) → memaksimalkan keinformatifan & mencegah bias Apriori.',
              'qcut gives balanced bins (~33% each) → maximising informativeness & preventing Apriori bias.')]),
        (t('Hasil & insight EDA', 'Results & EDA insight'), [
            t('Penipuan hanya di TRANSFER & CASH_OUT; DEBIT/CASH_IN/PAYMENT praktis 0 → fokus pengawasan menyempit.',
              'Fraud only in TRANSFER & CASH_OUT; DEBIT/CASH_IN/PAYMENT effectively 0 → supervision focus narrows.'),
            t('Banyak errorBalance ≠ 0 (artefak simulasi) → sinyal fitur, bukan alasan membuang baris.',
              'Many errorBalance ≠ 0 (simulation artefact) → a feature signal, not a reason to drop rows.'),
            t('Sampling 100k terbukti representatif terhadap populasi (uji KS ≈ 0,004 — distribusi hampir identik).',
              '100k sampling proven representative of the population (KS ≈ 0.004 — near-identical distributions).')]),
    ])
    P2 = _phase('Fase 2' if LANG == 'id' else 'Phase 2', 'Segmentation via Clustering', [
        (t('Tujuan', 'Goal'), [t('Menemukan pengelompokan alami perilaku transaksi & memberi nama bisnis tiap segmen.',
                                  'Find natural groupings of transaction behaviour & give each segment a business name.')]),
        (t('Yang dikerjakan', 'What we did'), [
            t('K-Means pada SELURUH 6,3 jt baris → 5 segmen bernama.', 'K-Means on ALL 6.3M rows → 5 named segments.'),
            t('DBSCAN (PCA-5D, sampel 50k) untuk deteksi noise/outlier; Hierarchical (sampel 5k, 3 linkage) untuk validasi struktur.',
              'DBSCAN (PCA-5D, 50k sample) for noise/outlier detection; Hierarchical (5k sample, 3 linkages) for structure validation.'),
            t('Profiling tiap cluster + ekspor label & jarak-ke-centroid untuk dipakai Phase 4.',
              'Per-cluster profiling + export of labels & distance-to-centroid for Phase 4.')]),
        (t('Metode & parameter', 'Method & parameters'), [
            t('K optimal = 5 (Elbow patah di K=5; Davies-Bouldin minimum 1,145; Silhouette plateau 0,367).',
              'Optimal K = 5 (Elbow knee at K=5; Davies-Bouldin minimum 1.145; Silhouette plateau 0.367).'),
            t('DBSCAN: eps dari k-distance (k=10) persentil-85, min_samples=10 → noise 8,7%.',
              'DBSCAN: eps from k-distance (k=10) 85th percentile, min_samples=10 → 8.7% noise.'),
            t('DBSCAN/Hierarchical disampel karena kompleksitas O(n²)/O(n³); K-Means (linear) dijalankan penuh.',
              'DBSCAN/Hierarchical sampled due to O(n²)/O(n³) cost; K-Means (linear) run in full.')]),
        (t('Justifikasi', 'Justification'), [
            t('Tiga algoritma = triangulasi 3 paradigma (centroid / densitas / koneksi); kesepakatan memperkuat validitas.',
              'Three algorithms = triangulating 3 paradigms (centroid / density / connectivity); agreement strengthens validity.'),
            t('K=5 dipilih atas dasar parsimoni + interpretabilitas, bukan sekadar argmax silhouette.',
              'K=5 chosen on parsimony + interpretability, not just argmax silhouette.')]),
        (t('Hasil & insight', 'Results & insight'), [
            t('Lima segmen bernama (lihat tab Segmentasi). Segmen "penguras saldo" & "penyusupan fraud" paling berisiko.',
              'Five named segments (see the Segmentation tab). The "balance-drainer" & "fraud-infiltration" segments are riskiest.'),
            t('Segmen paling berbahaya justru terlihat paling normal → alasan clustering + anomali dibutuhkan.',
              'The most dangerous segment looks the most normal → why clustering + anomaly detection are needed.')]),
    ])
    P3 = _phase('Fase 3' if LANG == 'id' else 'Phase 3', 'Association Rule Mining', [
        (t('Tujuan', 'Goal'), [t('Menemukan pola co-occurrence non-trivial antar atribut — 100% tanpa label.',
                                  'Find non-trivial co-occurrence patterns between attributes — 100% label-free.')]),
        (t('Yang dikerjakan', 'What we did'), [
            t('One-hot 7 atribut kategorikal → basket 20 item.', 'One-hot 7 categorical attributes → 20-item basket.'),
            t("Cek redundansi antar-atribut dengan Cramér's V (bukan Pearson, karena kategorikal).",
              "Inter-attribute redundancy check via Cramér's V (not Pearson, since categorical)."),
            t('Apriori → frequent itemsets → aturan + Support/Confidence/Lift → filter statistik + anti-tautologi → 12 aturan.',
              'Apriori → frequent itemsets → rules + Support/Confidence/Lift → statistical + anti-tautology filters → 12 rules.')]),
        (t('Metode & parameter', 'Method & parameters'), [
            t('min_support=0,01; max_len=4; low_memory=True — wajib pada skala 6,3 jt × 20 item agar tidak MemoryError (3,4 GiB).',
              'min_support=0.01; max_len=4; low_memory=True — required at 6.3M × 20 items to avoid MemoryError (3.4 GiB).'),
            t('Filter: lift>1,2; confidence>0,5; support>0,01; consequent tunggal; buang pasangan redundan.',
              'Filter: lift>1.2; confidence>0.5; support>0.01; single consequent; drop redundant pairs.')]),
        (t('Justifikasi', 'Justification'), [
            t('Ranking pakai Lift, bukan Confidence — Confidence menipu untuk item yang memang umum.',
              'Rank by Lift, not Confidence — Confidence is misleading for inherently common items.'),
            t("Anti-tautologi: type↔dest_kind redundan sempurna (Cramér's V = 1,00); emptied ⊂ drain (V = 0,82) — dibuang.",
              "Anti-tautology: type↔dest_kind perfectly redundant (Cramér's V = 1.00); emptied ⊂ drain (V = 0.82) — dropped.")]),
        (t('Hasil & insight', 'Results & insight'), [
            t('1.060 frequent itemsets → 7.242 kandidat → 477 strong rules → 12 aturan terdokumentasi.',
              '1,060 frequent itemsets → 7,242 candidates → 477 strong rules → 12 documented rules.'),
            t('Insight utama: perilaku pengurasan saldo MENGIKAT channel ke nominal — transfer/cash-out yang mengosongkan rekening hampir pasti bernilai besar (aturan 6 & 7).',
              'Key insight: balance-draining behaviour TIES channel to amount — transfers/cash-outs that empty accounts are almost always large (rules 6 & 7).')]),
    ])
    P4 = _phase('Fase 4' if LANG == 'id' else 'Phase 4', 'Anomaly & Outlier Detection', [
        (t('Tujuan', 'Goal'), [t('Menemukan record menyimpang & mengklasifikasi: data error / rare-but-legitimate / risk signal.',
                                  'Find deviating records & classify them: data error / rare-but-legitimate / risk signal.')]),
        (t('Yang dikerjakan', 'What we did'), [
            t('IQR + Z-score pada fitur MENTAH (agar interpretable) + Isolation Forest pada fitur ter-scaling.',
              'IQR + Z-score on RAW features (for interpretability) + Isolation Forest on scaled features.'),
            t('Ensemble voting 0–3; high-confidence = ≥2 metode setuju. Cross-reference dengan cluster outlier Phase 2.',
              'Ensemble voting 0–3; high-confidence = ≥2 methods agree. Cross-referenced with Phase 2 cluster outliers.'),
            t('Pola domain (complete-drain, night+high). Validasi post-hoc dengan isFraud, lalu ekspor anomaly report.',
              'Domain patterns (complete-drain, night+high). Post-hoc validation with isFraud, then export the anomaly report.')]),
        (t('Metode & parameter', 'Method & parameters'), [
            t('IQR multiplier 3,0; Z-score threshold 3σ; Isolation Forest: contamination 0,01, max_samples 256, 100 trees, seed 42.',
              'IQR multiplier 3.0; Z-score threshold 3σ; Isolation Forest: contamination 0.01, max_samples 256, 100 trees, seed 42.'),
            t('Cakupan sengaja berbeda: IQR ~43% (longgar), Z-score ~4,4%, Isolation Forest 1% (ketat, multivariat).',
              'Coverage intentionally differs: IQR ~43% (loose), Z-score ~4.4%, Isolation Forest 1% (strict, multivariate).')]),
        (t('Justifikasi', 'Justification'), [
            t('IQR 3,0 (bukan 1,5) karena variasi finansial besar → menekan false-positive.',
              'IQR 3.0 (not 1.5) because financial variation is large → reduces false positives.'),
            t('Isolation Forest dipilih (bukan LOF/One-Class SVM) karena skalabel untuk jutaan baris & multivariat.',
              'Isolation Forest chosen (over LOF/One-Class SVM) for scalability on millions of rows & multivariate handling.'),
            t('isFraud hanya disentuh di validasi akhir — bukan target mining.', 'isFraud touched only at final validation — never a mining target.')]),
        (t('Hasil & insight', 'Results & insight'), [
            t(f'Fraud rate naik monoton 0,05% → ± {v3:.1f}% seiring kesepakatan metode (lift ± 62× saat 3 metode setuju); ROC-AUC {kpis.get("auc", 0.946)}; 10% paling anomali memuat ~{kpis.get("top_decile_recall", 79)}% penipuan.',
              f'Fraud rate rises monotonically 0.05% → ± {v3:.1f}% with method agreement (lift ± 62× when 3 agree); ROC-AUC {kpis.get("auc", 0.946)}; top 10% anomalous holds ~{kpis.get("top_decile_recall", 79)}% of fraud.'),
            t('Insight kritis: anomali ≠ penipuan — DEBIT paling sering ditandai tapi 0 kasus penipuan (rare-but-legitimate). Fraud CASH_OUT justru menyamar → butuh ensemble + cross-reference.',
              'Critical insight: anomalous ≠ fraud — DEBIT is flagged most yet has 0 fraud (rare-but-legitimate). CASH_OUT fraud disguises itself → needs ensemble + cross-reference.')]),
    ])
    P5 = _phase('Fase 5' if LANG == 'id' else 'Phase 5', 'Visualization & Knowledge Presentation', [
        (t('Tujuan', 'Goal'), [t('Mengomunikasikan temuan ke audiens non-teknis lewat dashboard interaktif yang cepat.',
                                  'Communicate findings to a non-technical audience via a fast interactive dashboard.')]),
        (t('Yang dikerjakan', 'What we did'), [
            t('Notebook phase5_prepare_data.ipynb: pra-agregasi data 6,3 jt → berkas kecil ber-dimensi HARI (± 0,6 MB).',
              'phase5_prepare_data.ipynb: pre-aggregates 6.3M data → small DAY-dimensioned files (~0.6 MB).'),
            t('Dashboard Dash: slider waktu di Segmentasi/Pola/Anomali, kartu JIKA→MAKA berbahasa bisnis, dan tabel jelajah klik-untuk-jelaskan.',
              'Dash dashboard: time slider on Segmentation/Patterns/Anomaly, business-language IF→THEN cards, and a click-to-explain explorer table.')]),
        (t('Justifikasi (pipelining & sampling)', 'Justification (pipelining & sampling)'), [
            t('Pra-agregasi + penjumlahan bin per hari → latency mendekati nol saat slider digeser (tak mengolah 6,3 jt di sisi pengguna).',
              'Pre-aggregation + per-day bin summation → near-zero latency when sliding (no 6.3M processing client-side).'),
            t('Rasio bersifat sample-invariant → angka dari data penuh tetap representatif saat rentang digeser.',
              'Ratios are sample-invariant → full-data figures stay representative as the range changes.'),
            t('Anomali & scatter disampel (1,2 jt & 8.000) demi kecepatan; representativeness dibuktikan dengan uji KS.',
              'Anomaly & scatter are sampled (1.2M & 8,000) for speed; representativeness proven with a KS test.')]),
        (t('Hasil & insight', 'Results & insight'), [
            t('Temuan actionable: fokuskan pengawasan pada transfer/cash-out yang MENGURAS rekening — bukan sekadar nominal besar — dan pisahkan "anomali" dari "penipuan" agar analis tidak kebanjiran false-positive.',
              'Actionable finding: focus supervision on transfers/cash-outs that DRAIN accounts — not merely large amounts — and separate "anomalous" from "fraud" so analysts aren\'t flooded with false positives.')]),
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
        html.Div(t('Laporan Akhir', 'Final Report'), className='eyebrow'),
        html.P(t('Penjabaran lengkap tiap fase: tujuan, langkah, metode & parameter, justifikasi, hasil & insight — sesuai keluaran notebook Phase 1–5.',
                 'Full breakdown of each phase: goal, steps, method & parameters, justification, results & insight — matching the Phase 1–5 notebook outputs.'), className='report-lead'),
        html.Hr(className='hr'), *blocks,
        html.Div(t([html.Span('Intinya: tanpa pernah memakai label penipuan saat mining, '),
                    html.Span('10%', className='hl'), html.Span(' transaksi paling janggal memuat sekitar '),
                    html.Span(f"{kpis.get('top_decile_recall', 76)}%", className='hl'),
                    html.Span(' seluruh penipuan (ketajaman '), html.Span(f"{kpis.get('auc', 0.94)}", className='hl'),
                    html.Span('). Di sanalah pengawasan sebaiknya difokuskan.')],
                   [html.Span('Bottom line: without ever using fraud labels during mining, '),
                    html.Span('10%', className='hl'), html.Span(' of the most anomalous transactions hold about '),
                    html.Span(f"{kpis.get('top_decile_recall', 76)}%", className='hl'),
                    html.Span(' of all fraud (sharpness '), html.Span(f"{kpis.get('auc', 0.94)}", className='hl'),
                    html.Span('). That is where supervision should focus.')]), className='report-quote'),
    ], className='report'))

# ════════════════════════════════════════════════════════════════════════════
# LAYOUT + NAVIGASI
# ════════════════════════════════════════════════════════════════════════════
app = Dash(__name__, suppress_callback_exceptions=True, title='PaySim KDD Console')
server = app.server

def page_not_ready():
    return html.Div([html.H2(t('Data belum disiapkan', 'Data not prepared yet'), className='page-title'),
                     callout(t('Jalankan notebook dulu', 'Run the notebook first'),
                             t('Folder ../data masih kosong. Jalankan seluruh sel notebooks/Phase5/phase5_prepare_data.ipynb (env DataMining), lalu muat ulang halaman ini.',
                               'The ../data folder is empty. Run all cells of notebooks/Phase5/phase5_prepare_data.ipynb (DataMining env), then reload this page.'), 'warn')], className='content')

if DATA_READY:
    app.layout = html.Div([dcc.Store(id='page', data='ov'), dcc.Store(id='lang', data='id'), sidebar,
                           html.Div([topbar, html.Div(id='content', className='content')], className='main')], className='app')
else:
    app.layout = html.Div([dcc.Store(id='lang', data='id'), sidebar,
                           html.Div([topbar, page_not_ready()], className='main')], className='app')

if DATA_READY:
    @app.callback(Output('page', 'data'), [Input(f'nav-{c}', 'n_clicks') for c, *_ in NAV], prevent_initial_call=True)
    def navigate(*_):
        return (ctx.triggered_id or 'nav-ov').replace('nav-', '')

    @app.callback([Output(f'nav-{c}', 'className') for c, *_ in NAV], Input('page', 'data'))
    def set_active(page):
        return ['nav-item active' if c == page else 'nav-item' for c, *_ in NAV]

    # ── Bahasa: set global LANG + terjemahkan chrome (nav + label latency) ──
    @app.callback([Output('lang', 'data')] + [Output(f'navlbl-{c}', 'children') for c, *_ in NAV] + [Output('lat-label', 'children')],
                  Input('lang-toggle', 'value'))
    def set_lang(v):
        global LANG
        LANG = v if v in ('id', 'en') else 'id'
        labels = [(enl if LANG == 'en' else idl) for _, idl, enl in NAV]
        return [LANG] + labels + [t('Latency', 'Latency')]

    @app.callback(Output('content', 'children'), Input('page', 'data'), Input('lang', 'data'))
    def render(page, _lang):
        comp = {'ov': page_overview, 'cl': page_segmentation, 'ru': page_rules,
                'an': page_anomaly, 'ins': page_insight, 'doc': page_doc}.get(page, page_overview)()
        bs = behind_scenes(page)
        if bs is not None:
            try:
                comp.children.append(bs)
            except Exception:
                pass
        return comp

    # ── Insight: simulator ──
    @app.callback(Output('sim-caught', 'children'), Output('sim-sentence', 'children'), Output('sim-fig', 'figure'),
                  Input('sim-slider', 'value'))
    def cb_sim(pct):
        if gains is None:
            return no_update, no_update, no_update
        caught = gains_lookup(pct)
        sentence = t(['Dengan memeriksa hanya ', html.B(f'{pct}%'), ' transaksi paling mencurigakan, tim menangkap sekitar ',
                      html.B(f'{caught:.0f}%'), ' dari SELURUH penipuan — sisa ', html.B(f'{100-pct}%'),
                      ' transaksi tak perlu disentuh. Efisiensi inilah nilai utama proyek ini.'],
                     ['By inspecting just ', html.B(f'{pct}%'), ' of the most suspicious transactions, the team catches about ',
                      html.B(f'{caught:.0f}%'), ' of ALL fraud — the remaining ', html.B(f'{100-pct}%'),
                      ' need not be touched. This efficiency is the core value of the project.'])
        return f'{caught:.0f}%', sentence, fig_gains(pct, caught)

    # ── Overview ──
    @app.callback(Output('ov-type', 'figure'), Input('ov-metric', 'value'))
    def cb_ov(metric):
        d = type_dist.sort_values(metric, ascending=False)
        y = 'count' if metric == 'count' else 'fraud_rate'
        f = go.Figure(go.Bar(x=d['type'], y=d[y], marker_color=INDIGO,
                             text=[f'{v:,.0f}' if metric == 'count' else f'{v:.3f}%' for v in d[y]], textposition='outside'))
        f = style_fig(f, 300)
        f.update_yaxes(title=t('Jumlah', 'Count') if metric == 'count' else t('Tingkat penipuan (%)', 'Fraud rate (%)'))
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
        lbl = t('Segmen', 'Segment')
        d[lbl] = d['cluster'].map(lambda c: f"C{c} · {CLUSTER_NAME.get(c, c)}")
        f = px.scatter(d, x='pc1', y='pc2', color=(lbl if colorby == 'cluster' else 'type'),
                       color_discrete_sequence=PALETTE, opacity=0.6)
        f.update_traces(marker=dict(size=5))
        f = style_fig(f, 430)
        f.update_layout(legend=dict(title='', orientation='v', font=dict(size=10.5)), xaxis_title='', yaxis_title='')
        return f

    @app.callback(Output('cl-table', 'children'), Output('cl-note', 'children'), Input('cl-days', 'value'))
    def cb_cltable(days):
        g = clusters_for_days(days[0], days[1]).sort_values('cluster')
        top = g.sort_values('fraud_rate').iloc[-1]
        note = [html.Div(t('💡 Catatan analis', '💡 Analyst note'), className='callout-title'),
                html.P(t([f"Pada hari {days[0]}–{days[1]}, segmen paling berisiko adalah ",
                          html.B(f"C{int(top.cluster)} ({top['name']})"),
                          f" dengan tingkat penipuan {top.fraud_rate:.3f}% dan rata-rata pengurasan {top.mean_drain:.1f}×. "
                          "Segmen inilah yang paling layak diprioritaskan untuk investigasi."],
                         [f"On days {days[0]}–{days[1]}, the riskiest segment is ",
                          html.B(f"C{int(top.cluster)} ({top['name']})"),
                          f" with a fraud rate of {top.fraud_rate:.3f}% and average drain {top.mean_drain:.1f}×. "
                          "This segment most deserves investigation priority."]))]
        return cluster_table(g), note

    @app.callback(Output('cd-detail', 'children'), Input('cd-pick', 'value'), Input('cl-days', 'value'))
    def cb_cldetail(c, days):
        g = clusters_for_days(days[0], days[1])
        row = g[g.cluster == c]
        if len(row) == 0:
            return html.P(t('Tidak ada data pada rentang ini.', 'No data in this range.'))
        row = row.iloc[0]
        story = CLUSTER_STORY.get(int(c))
        if story:
            title, karak, insight, impl = pick(story)
        else:
            title, karak, insight, impl = CLUSTER_NAME.get(c, f'Segment {c}'), '', '', ''
        comp = cluster_comp(c, days[0], days[1])
        def stat(lbl, val):
            return html.Div([html.Div(lbl, className='kpi-label'),
                             html.Div(val, className='kpi-val', style={'fontSize': '18px'})], className='kpi')
        stats = html.Div([stat(t('Jumlah', 'Count'), f"{int(row['n']):,}"), stat('% Data', f"{row['pct']:.1f}%"),
                          stat(t('Rata Nominal', 'Avg Amount'), f"{int(row['mean_amount']):,}"), stat(t('Rata Penguras', 'Avg Drain'), f"{row['mean_drain']:.1f}×"),
                          stat(t('Penipuan', 'Fraud'), f"{row['fraud_rate']:.3f}%")], className='kpi-row', style={'marginTop': '10px'})
        return html.Div([
            html.H3(title, style={'margin': '4px 0 2px'}),
            html.Div(t(f"Komposisi jenis (rentang ini): {comp}", f"Type composition (this range): {comp}"), className='caption'),
            stats,
            html.Div([html.Div(t('KARAKTERISTIK', 'CHARACTERISTICS'), className='fh karak'), html.Div(karak, className='fb')], className='story-field'),
            html.Div([html.Div('INSIGHT', className='fh insight'), html.Div(insight, className='fb')], className='story-field'),
            html.Div([html.Div(t('IMPLEMENTASI', 'IMPLEMENTATION'), className='fh impl'), html.Div(impl, className='fb')], className='story-field'),
        ])

    # ── Rules ──
    @app.callback(Output('ru-network', 'figure'), Output('ru-cards', 'children'), Output('ru-count', 'children'),
                  Input('ru-lift', 'value'), Input('ru-days', 'value'))
    def cb_rules(min_lift, days):
        r = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False)
        cards = html.Div([rule_card(row, days[0], days[1]) for _, row in r.iterrows()], className='rule-grid')
        tot = range_total_tx(days[0], days[1])
        warn = t(' ⚠ Rentang sangat kecil — angka aktivitas bisa tidak stabil.', ' ⚠ Very small range — activity figures may be unstable.') if tot < 20000 else ''
        msg = [html.B(t(f'{len(r)} pola', f'{len(r)} patterns')),
               t(f' ditampilkan (dari {len(rules)}). Total {tot:,} transaksi pada hari {days[0]}–{days[1]}.',
                 f' shown (of {len(rules)}). Total {tot:,} transactions on days {days[0]}–{days[1]}.'),
               html.B(warn, style={'color': AMBER})]
        return fig_rule_network(min_lift), cards, msg

    # ── Anomaly ──
    @app.callback(Output('an-main', 'figure'), Output('an-caption', 'children'), Output('an-key', 'children'),
                  Output('an-votes', 'children'), Input('an-view', 'value'), Input('an-days', 'value'))
    def cb_anom(view, days):
        vv = votes_for_days(days[0], days[1]).sort_values('vote')
        try:
            v3 = float(vv.loc[vv['vote'] == 3, 'fraud_rate'].iloc[0]); mult = v3 / max(kpis['fraud_rate'], 1e-9)
            key = [html.Div(t('💡 Temuan kunci', '💡 Key finding'), className='callout-title'),
                   html.P(t([f"Saat KETIGA metode setuju, tingkat penipuan melonjak jadi ", html.B(f"{v3:.1f}%"),
                             f" (± {mult:.0f}× lipat rata-rata) — ", html.B('ditemukan tanpa memakai label'), "."],
                            [f"When ALL THREE methods agree, the fraud rate jumps to ", html.B(f"{v3:.1f}%"),
                             f" (± {mult:.0f}× the average) — ", html.B('found without using labels'), "."]))]
        except Exception:
            key = [html.Div(t('💡 Temuan kunci', '💡 Key finding'), className='callout-title'), html.P(t('Data rentang tidak cukup.', 'Not enough data in range.'))]
        vlist = [html.Div(t('SUARA "JANGGAL" (0–3 metode)', 'ANOMALY VOTES (0–3 methods)'), className='kpi-label', style={'marginBottom': '6px'})]
        for r in vv.itertuples():
            vlist.append(html.Div([html.Span(f"{int(r.vote)} {t('metode','methods')}"), html.Span(f"{r.pct:.1f}%", className='v')], className='mini-row'))

        if view == 'vote':
            f = go.Figure()
            f.add_bar(x=vv['vote'], y=vv['pct'], name=t('% transaksi', '% of transactions'), marker_color='#D8DBF2')
            f.add_scatter(x=vv['vote'], y=vv['fraud_rate'], name=t('Tingkat penipuan (%)', 'Fraud rate (%)'), mode='lines+markers',
                          line=dict(color=RED, width=4), marker=dict(size=9), yaxis='y2')
            f = style_fig(f, 420)
            f.update_layout(yaxis=dict(title=t('% transaksi', '% of transactions')),
                            yaxis2=dict(title=t('Penipuan (%)', 'Fraud (%)'), overlaying='y', side='right', showgrid=False),
                            legend=dict(orientation='h', y=1.1), xaxis=dict(title=t('Jumlah metode yang setuju (0–3)', 'Methods in agreement (0–3)'), dtick=1))
            cap = [t('Garis merah naik terus ke kanan: makin banyak metode sepakat sebuah transaksi janggal, makin besar peluang penipuan.',
                     'The red line keeps rising to the right: the more methods agree a transaction is anomalous, the higher the fraud probability.')]
        else:
            key_col = 'type' if view == 'type' else 'cluster'
            g = anom_group(an_type if view == 'type' else an_clu, key_col, days[0], days[1])
            xc = g['type'] if view == 'type' else ('C' + g['cluster'].astype(str))
            f = go.Figure()
            f.add_bar(x=xc, y=g['anomaly_rate'], name=t('Tingkat janggal (%)', 'Anomaly rate (%)'), marker_color=AMBER)
            f.add_bar(x=xc, y=g['fraud_rate'], name=t('Tingkat penipuan (%)', 'Fraud rate (%)'), marker_color=RED)
            f = style_fig(f, 420)
            f.update_layout(barmode='group', yaxis_title='%', legend=dict(orientation='h', y=1.1))
            cap = [t('Batang oranye tinggi tapi merah pendek = "sering janggal, jarang penipuan" (mis. DEBIT: langka tapi sah).',
                     'Tall amber but short red = "often anomalous, rarely fraud" (e.g. DEBIT: rare but legitimate).')]
        return f, cap, key, html.Div(vlist)

    @app.callback(Output('ex-dt', 'data'), Output('ex-count', 'children'),
                  Input('ex-filter', 'value'), Input('ex-type', 'value'), Input('an-days', 'value'))
    def cb_ex(filt, typ, days):
        d = explorer[(explorer['day'] >= days[0]) & (explorer['day'] <= days[1])]
        if filt == 'fraud':   d = d[d['fraud'] == 1]
        elif filt == 'anom':  d = d[d['anomali'] == 1]
        elif filt == 'normal': d = d[(d['anomali'] == 0) & (d['fraud'] == 0)]
        if typ != 'all':      d = d[d['type'] == typ]
        msg = t([f"Menampilkan {len(d):,} transaksi (hari {days[0]}–{days[1]}). ", html.B('Klik satu baris'), ' untuk penjelasan; ketik di kotak filter kolom untuk mencari.'],
                [f"Showing {len(d):,} transactions (days {days[0]}–{days[1]}). ", html.B('Click a row'), ' to explain; type in a column filter box to search.'])
        return d.to_dict('records'), msg

    @app.callback(Output('ex-explain', 'children'), Input('ex-dt', 'active_cell'), State('ex-dt', 'derived_viewport_data'))
    def cb_explain(active, vdata):
        # derived_viewport_data = baris pada HALAMAN yang sedang tampil; active_cell['row'] indeks page-relatif.
        if not active or not vdata or active['row'] >= len(vdata):
            return explain_placeholder()
        r = vdata[active['row']]
        methods = []
        if r.get('is_iqr'):  methods.append(t('IQR — nilainya jauh di luar rentang wajar (kuartil)', 'IQR — value far outside the reasonable range (quartiles)'))
        if r.get('is_z'):    methods.append(t('Z-score — lebih dari 3 simpangan baku dari rata-rata', 'Z-score — more than 3 standard deviations from the mean'))
        if r.get('is_iso'):  methods.append(t('Isolation Forest — kombinasi ciri-cirinya langka sehingga mudah "diisolasi"', 'Isolation Forest — its feature combination is rare, so easily "isolated"'))
        vote = int(r.get('anomaly_vote', 0))
        head = t(f"Transaksi {r['type']} · nominal {int(r['amount']):,} · hari {int(r['day'])}",
                 f"{r['type']} transaction · amount {int(r['amount']):,} · day {int(r['day'])}")
        if vote == 0:
            body = [html.P([html.B(t('Bukan anomali. ', 'Not anomalous. ')), t('Tidak ada metode yang menandai transaksi ini — wajar dari semua sisi.',
                                                                                'No method flagged this transaction — reasonable on every axis.')])]
        else:
            body = [html.P([html.B(t(f'Ditandai janggal oleh {vote} dari 3 metode:', f'Flagged anomalous by {vote} of 3 methods:'))]),
                    html.Ul([html.Li(m) for m in methods], style={'margin': '2px 0 8px', 'paddingLeft': '20px'}),
                    html.P([html.B(t('Pemicu utama: ', 'Main trigger: ')), r.get('top_reason', '-'), '.']),
                    html.P([html.B(t('Konteks segmen: ', 'Segment context: ')),
                            t(f"masuk Segmen {int(r['cluster'])} ({r['segmen']}), berjarak {r.get('dist', 0)} dari pusat segmennya ",
                              f"in Segment {int(r['cluster'])} ({r['segmen']}), {r.get('dist', 0)} away from its segment centre ")
                            + (t('— cukup jauh, menyimpang dari perilaku normal segmennya.', '— quite far, deviating from its segment\'s normal behaviour.') if r.get('dist', 0) > 3
                               else t('— relatif dekat pusat, jadi janggal karena nilainya ekstrem, bukan posisinya.', '— relatively close to the centre, so anomalous due to extreme values, not position.'))])]
        verdict = (t('Ternyata PENIPUAN (dicek dari label — hanya untuk validasi).', 'Turns out FRAUD (checked from label — validation only).') if r.get('fraud')
                   else t('Ternyata BUKAN penipuan (validasi) — bukti bahwa "janggal" tak selalu berarti "penipuan".', 'Turns out NOT fraud (validation) — proof that "anomalous" ≠ "fraud".'))
        return html.Div([html.Div('🔎 ' + head, className='callout-title')] + body +
                        [html.P(verdict, style={'marginTop': '6px', 'fontWeight': 600,
                                                'color': (RED if r.get('fraud') else TEAL)})])

if __name__ == '__main__':
    app.run(debug=False, port=8050)
