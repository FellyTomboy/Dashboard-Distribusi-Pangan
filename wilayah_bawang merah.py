#!/usr/bin/env python3
"""
Analisis Spasial Prediksi Harga - Bawang Merah
===============================================
Klasifikasi wilayah harga tinggi/harga rendah/harga rata-rata berdasarkan perbandingan
harga prediksi tiap kabupaten/kota terhadap rata-rata provinsi.

LOGIKA EKONOMI:
  - Harga TINGGI di suatu wilayah -> pasokan KURANG -> harga tinggi -> butuh distribusi
  - Harga RENDAH di suatu wilayah -> pasokan LEBIH -> harga rendah -> bisa jadi sumber distribusi
  - Harga di sekitar rata-rata provinsi -> harga rata-rata

Visualisasi:
  1. Matriks klasifikasi harga tinggi/harga rendah per wilayah x horizon (H+1 - H+7)
  2. Peta konsistensi spasial - dominasi harga tinggi/harga rendah tiap kabupaten/kota
  3. Peta ketimpangan harga Jawa Timur per horizon
  4. Analisis temporal ketimpangan di sekitar hari libur (Lebaran, Nataru)

Data: test_predictions_bawang_merah.csv (periode uji 2024-2025)
Output: folder 'bawang merah/analisis prediksi/output/'
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
KOMODITAS = "Bawang Merah"
KOMODITAS_TAG = "bawang_merah"

PRED_PATH = "bawang merah/training/test_predictions_bawang_merah.csv"
GEOJSON_PATH = "Data Lain-Lain/all_kabkota_ind.geojson"
KOORD_PATH = "Data Lain-Lain/koordinat_jatim.csv"
LIBUR_PATH = "Data Lain-Lain/libur_nasional_2018-2025.csv"

OUTPUT_DIR = "bawang merah/analisis prediksi/output"

# Rentang peta Jawa Timur
JATIM_LON_RANGE = (110.5, 115.0)
JATIM_LAT_RANGE = (-8.9, -6.7)

# Metode threshold: 'empiris' (persentil 25/75) atau 'fixed'
THRESHOLD_METHOD = 'empiris'

# Jika fixed: +/- 15%
FIXED_THRESHOLD = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})


# ============================================================
# FUNGSI PEMBANTU
# ============================================================
def normalise_nama(nama: str) -> str:
    """Seragamkan nama wilayah: uppercase, strip, hapus spasi ganda."""
    return ' '.join(nama.strip().upper().split())


def match_prediction_to_geojson(nama_pred: str, geojson_lookup: dict):
    """Cocokkan nama prediction CSV ke alt_name GeoJSON."""
    key = normalise_nama(nama_pred)
    if key in geojson_lookup:
        return key
    for prefix in ('KABUPATEN ', 'KOTA '):
        if key.startswith(prefix):
            stripped = key[len(prefix):]
            for gk in geojson_lookup:
                if gk == stripped:
                    return gk
    return None


def extract_polygon_coords(geometry):
    """Ekstrak koordinat (lon, lat) dari GeoJSON Polygon / MultiPolygon."""
    coords_list = []
    if geometry['type'] == 'Polygon':
        coords_list.append(np.array(geometry['coordinates'][0]))
    elif geometry['type'] == 'MultiPolygon':
        for poly in geometry['coordinates']:
            coords_list.append(np.array(poly[0]))
    return coords_list


def classify_market(pct_diff: float, lower: float, upper: float) -> str:
    """
    Klasifikasi kondisi pasar berdasarkan deviasi harga terhadap rata-rata provinsi.

    Parameters
    ----------
    pct_diff : (harga_kab - harga_provinsi) / harga_provinsi
    lower    : threshold bawah (harga rendah jika di bawah ini)
    upper    : threshold atas  (harga tinggi jika di atas ini)
    """
    if pct_diff > upper:
        return 'Harga Tinggi'
    elif pct_diff < lower:
        return 'Harga Rendah'
    else:
        return 'Harga Rata-rata'


def get_holiday_periods(df_libur: pd.DataFrame) -> list:
    """Identifikasi periode Lebaran dan Nataru. Return [(start, end, label, color, alpha)]."""
    periods = []
    # Lebaran: Idul Fitri +/- 7 hari
    idul_fitri = df_libur[
        df_libur['hari_raya'].str.contains('idul fitri|idulfitri', case=False, na=False)
        & ~df_libur['hari_raya'].str.contains('cuti bersama', case=False, na=False)
    ].copy()
    for _, row in idul_fitri.iterrows():
        t = pd.Timestamp(row['tanggal'])
        year = t.year
        periods.append((t - timedelta(days=7), t + timedelta(days=7),
                        f'Lebaran {year}', '#e74c3c', 0.3))
    # Nataru: 20 Des - 5 Jan
    for year in range(2018, 2026):
        start = pd.Timestamp(f'{year}-12-20')
        end = pd.Timestamp(f'{year + 1}-01-05')
        if start >= pd.Timestamp('2018-01-01') and end <= pd.Timestamp('2025-12-31'):
            periods.append((start, end, f'Nataru {year}/{year+1}', '#3498db', 0.3))
    return periods


# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print(f"  Analisis Spasial - {KOMODITAS}")
print("=" * 60)

print("\n[1] Memuat data prediksi...")
df_pred = pd.read_csv(PRED_PATH, parse_dates=['forecast_date'])
print(f"     -> {len(df_pred):,} baris, {df_pred['kab_kota'].nunique()} wilayah, "
      f"horizon {df_pred['horizon'].min()}-{df_pred['horizon'].max()}, "
      f"periode {df_pred['forecast_date'].min():%Y-%m} - {df_pred['forecast_date'].max():%Y-%m}")

# Filter: hapus baris yang horizonnya tidak lengkap (< 7 per tanggal+kab)
n_before = len(df_pred)
horizon_count = df_pred.groupby(['kab_kota', 'forecast_date'])['horizon'].nunique().reset_index()
complete_dates = horizon_count[horizon_count['horizon'] == 7]
df_pred = df_pred.merge(
    complete_dates[['kab_kota', 'forecast_date']],
    on=['kab_kota', 'forecast_date'],
    how='inner'
)
print(f"     -> Filter horizon: {n_before} -> {len(df_pred)} rows "
      f"({n_before - len(df_pred)} removed, "
      f"{(1 - len(df_pred)/n_before)*100:.1f}% incomplete)")

print("\n[2] Memuat data hari libur...")
df_libur = pd.read_csv(LIBUR_PATH, parse_dates=['tanggal'])
print(f"     -> {len(df_libur)} hari libur (2018-2025)")

print("\n[3] Memuat GeoJSON & filter Jawa Timur...")
with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
    geojson = json.load(f)

jatim_features = [
    f for f in geojson['features']
    if f['properties'].get('prov_name', '').upper() == 'JAWA TIMUR'
]
print(f"     -> {len(jatim_features)} fitur Jawa Timur")

feature_lookup = {}
for feat in jatim_features:
    alt_name = normalise_nama(feat['properties']['alt_name'])
    feature_lookup[alt_name] = feat

pred_names = df_pred['kab_kota'].unique()
name_mapping = {}
unmatched = []
for pname in pred_names:
    matched = match_prediction_to_geojson(pname, feature_lookup)
    if matched:
        name_mapping[pname] = matched
    else:
        unmatched.append(pname)

print(f"     -> Tercocokkan: {len(name_mapping)} / {len(pred_names)} wilayah")
if unmatched:
    print(f"     -> !!! Tidak tercocokkan: {unmatched}")

df_pred['alt_name'] = df_pred['kab_kota'].map(name_mapping)


# ============================================================
# 2. KLASIFIKASI HARGA TINGGI / HARGA RENDAH / HARGA RATA-RATA
# ============================================================
print("\n[4] Klasifikasi harga tinggi/harga rendah berdasarkan harga relatif...")

# --- Hitung rata-rata provinsi per (tanggal, horizon) ---
rata_provinsi = (
    df_pred.groupby(['forecast_date', 'horizon'])['prediction']
    .mean()
    .reset_index()
    .rename(columns={'prediction': 'rata_provinsi_pred'})
)
rata_provinsi_actual = (
    df_pred.groupby(['forecast_date', 'horizon'])['actual']
    .mean()
    .reset_index()
    .rename(columns={'actual': 'rata_provinsi_actual'})
)
df_pred = df_pred.merge(rata_provinsi, on=['forecast_date', 'horizon'], how='left')
df_pred = df_pred.merge(rata_provinsi_actual, on=['forecast_date', 'horizon'], how='left')

# --- Deviasi harga tiap kab terhadap rata-rata provinsi ---
df_pred['pct_vs_provinsi_pred'] = (
    (df_pred['prediction'] - df_pred['rata_provinsi_pred']) / df_pred['rata_provinsi_pred']
)
df_pred['pct_vs_provinsi_actual'] = (
    (df_pred['actual'] - df_pred['rata_provinsi_actual']) / df_pred['rata_provinsi_actual']
)

# --- Tentukan threshold ---
all_pct = df_pred['pct_vs_provinsi_pred'].dropna()
if THRESHOLD_METHOD == 'empiris':
    lower_thresh = all_pct.quantile(0.25)
    upper_thresh = all_pct.quantile(0.75)
    print(f"\n  Threshold empiris (persentil 25/75):")
    print(f"     Harga Rendah (bawah): {lower_thresh:.2%}")
    print(f"     Harga Tinggi (atas) : {upper_thresh:.2%}")
    print(f"     Artinya: 25% termurah -> Harga Rendah, 25% termahal -> Harga Tinggi, 50% tengah -> Harga Rata-rata")
else:
    lower_thresh = -FIXED_THRESHOLD
    upper_thresh = FIXED_THRESHOLD
    print(f"\n  Threshold fixed: +/-{FIXED_THRESHOLD:.0%}")

# --- Klasifikasi ---
df_pred['klasifikasi_pred'] = df_pred['pct_vs_provinsi_pred'].apply(
    lambda x: classify_market(x, lower_thresh, upper_thresh)
)
df_pred['klasifikasi_actual'] = df_pred['pct_vs_provinsi_actual'].apply(
    lambda x: classify_market(x, lower_thresh, upper_thresh)
)
df_pred['class_code_pred'] = df_pred['klasifikasi_pred'].map({'Harga Rendah': -1, 'Harga Rata-rata': 0, 'Harga Tinggi': 1})
df_pred['class_code_actual'] = df_pred['klasifikasi_actual'].map({'Harga Rendah': -1, 'Harga Rata-rata': 0, 'Harga Tinggi': 1})

# --- Distribusi klasifikasi ---
for src, label in [('pred', '(berdasarkan PREDIKSI)'), ('actual', '(berdasarkan AKTUAL)')]:
    counts = df_pred[f'klasifikasi_{src}'].value_counts()
    total = counts.sum()
    print(f"\n  Distribusi klasifikasi {label}:")
    for cat in ['Harga Rendah', 'Harga Rata-rata', 'Harga Tinggi']:
        n = counts.get(cat, 0)
        print(f"     {cat:10s} : {n:>8,} ({n / total:.1%})")

# --- Agreement prediksi vs aktual ---
agreement = (df_pred['klasifikasi_pred'] == df_pred['klasifikasi_actual']).mean()
print(f"\n  Agreement prediksi vs aktual: {agreement:.1%}")
print(f"  (seberapa sering klasifikasi dari PREDIKSI cocok dengan AKTUAL)")


# ============================================================
# AGREGASI PER WILAYAH
# ============================================================
print("\n[5] Agregasi per wilayah...")

# Rata-rata pct_vs_provinsi per wilayah + horizon
mean_deviasi = df_pred.groupby(['alt_name', 'horizon'], as_index=False).agg(
    deviasi_pred=('pct_vs_provinsi_pred', 'mean'),
    deviasi_actual=('pct_vs_provinsi_actual', 'mean'),
)

# Dominant classification per wilayah (PREDIKSI)
dom_series = df_pred.groupby('alt_name')['klasifikasi_pred'].value_counts(normalize=True).mul(100).round(1)
dominant = dom_series.reset_index()
dominant.columns = ['alt_name', 'klasifikasi', 'persentase']
dominant_class = dominant.loc[dominant.groupby('alt_name')['persentase'].idxmax()].reset_index(drop=True)
dominant_class['keyakinan'] = dominant_class['persentase'] / 100.0

# Dominant classification per wilayah (AKTUAL)
dom_series_act = df_pred.groupby('alt_name')['klasifikasi_actual'].value_counts(normalize=True).mul(100).round(1)
dominant_act = dom_series_act.reset_index()
dominant_act.columns = ['alt_name', 'klasifikasi', 'persentase']
dominant_class_act = dominant_act.loc[dominant_act.groupby('alt_name')['persentase'].idxmax()].reset_index(drop=True)

# Agreement per wilayah
wilayah_agree = df_pred.groupby('alt_name').apply(
    lambda g: (g['klasifikasi_pred'] == g['klasifikasi_actual']).mean()
).reset_index().rename(columns={0: 'agreement'})

# ============================================================
# METRIK KETIMPANGAN HARGA
# ============================================================
print("\n[6] Menghitung metrik ketimpangan harga...")

# CoV (Coefficient of Variation) per tanggal + horizon
ketimpangan = (
    df_pred.groupby(['forecast_date', 'horizon'])['prediction']
    .agg(['mean', 'std'])
    .reset_index()
)
ketimpangan['cov'] = ketimpangan['std'] / ketimpangan['mean'].clip(lower=1)
ketimpangan['cov_pct'] = ketimpangan['cov'] * 100

# Rata-rata CoV per horizon
cov_per_horizon = ketimpangan.groupby('horizon')['cov_pct'].agg(['mean', 'std']).reset_index()

# Same but for actual values
ketimpangan_actual = (
    df_pred.groupby(['forecast_date', 'horizon'])['actual']
    .agg(['mean', 'std'])
    .reset_index()
)
ketimpangan_actual['cov'] = ketimpangan_actual['std'] / ketimpangan_actual['mean'].clip(lower=1)
ketimpangan_actual['cov_pct'] = ketimpangan_actual['cov'] * 100

print(f"     CoV rata-rata antar horizon: "
      f"{cov_per_horizon['mean'].mean():.2f}% (prediksi) / "
      f"{ketimpangan_actual.groupby('horizon')['cov_pct'].mean().mean():.2f}% (aktual)")


# ============================================================
# FIGURE 1: MATRIKS KLASIFIKASI (wilayah x horizon)
# ============================================================
print("\n[7] Figure 1: Matriks klasifikasi...")

pivot_deviasi = mean_deviasi.pivot(index='alt_name', columns='horizon', values='deviasi_pred')
pivot_deviasi = pivot_deviasi[sorted(pivot_deviasi.columns)]

fig1, ax1 = plt.subplots(figsize=(14, 10))

# Diverging colormap: merah (harga tinggi) -> putih -> hijau (harga rendah)
cmap_div = LinearSegmentedColormap.from_list('harga_tinggi_rendah',
    ['#1a9641', '#f7f7f7', '#d73027'], N=256)
vmax = max(abs(pivot_deviasi.values.min()), abs(pivot_deviasi.values.max()))

im = ax1.imshow(pivot_deviasi.values, aspect='auto', cmap=cmap_div,
                vmin=-vmax, vmax=vmax, interpolation='nearest')

# Anotasi
for i in range(pivot_deviasi.shape[0]):
    for j in range(pivot_deviasi.shape[1]):
        val = pivot_deviasi.iloc[i, j]
        if val > upper_thresh:
            sym, c = '-', 'white'
        elif val < lower_thresh:
            sym, c = '+', 'white'
        else:
            sym, c = '.', '#666666'
        ax1.text(j, i, sym, ha='center', va='center', fontsize=7,
                fontweight='bold', color=c)

ax1.set_xticks(range(len(pivot_deviasi.columns)))
ax1.set_xticklabels([f'H+{h}' for h in pivot_deviasi.columns], fontsize=10, fontweight='bold')
yticklabels = [nm.replace('KABUPATEN ', '').replace('KOTA ', '').title()
               for nm in pivot_deviasi.index]
ax1.set_yticks(range(len(pivot_deviasi.index)))
ax1.set_yticklabels(yticklabels, fontsize=8)
ax1.set_xlabel('Horizon Prediksi', fontsize=12, fontweight='bold')
ax1.set_ylabel('Kabupaten / Kota', fontsize=12, fontweight='bold')
ax1.set_title(
    f'Klasifikasi Harga Tinggi / Harga Rendah per Wilayah x Horizon\n'
    f'{KOMODITAS}  |  + = Harga Rendah  - = Harga Tinggi',
    fontsize=13, fontweight='bold', pad=12)

cbar = plt.colorbar(im, ax=ax1, shrink=0.6, pad=0.015)
cbar.set_label('Deviasi harga dari rata-rata provinsi', fontsize=10)
cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, '01_klasifikasi_matrix.png'), dpi=200, bbox_inches='tight')
plt.close(fig1)
print("     OK -> 01_klasifikasi_matrix.png")


# ============================================================
# FIGURE 2: PETA KONSISTENSI SPASIAL
# ============================================================
print("\n[8] Figure 2: Peta konsistensi spasial...")

class_to_num = {'Harga Rendah': -1, 'Harga Rata-rata': 0, 'Harga Tinggi': 1}
dominant_dict = dict(zip(dominant_class['alt_name'],
                         dominant_class['klasifikasi'].map(class_to_num)))
confidence_dict = dict(zip(dominant_class['alt_name'], dominant_class['keyakinan']))

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 8))

cmap_class = LinearSegmentedColormap.from_list('class_cmap2',
    ['#1a9641', '#f7f7f7', '#d73027'], N=256)

# Panel kiri: klasifikasi dominan
for feat in jatim_features:
    alt_name = normalise_nama(feat['properties']['alt_name'])
    val = dominant_dict.get(alt_name, None)
    if val is not None:
        color = cmap_class((val + 1) / 2)
    else:
        color = '#e0e0e0'
    for coords in extract_polygon_coords(feat['geometry']):
        poly = MplPolygon(coords, closed=True, facecolor=color,
                          edgecolor='white', linewidth=0.4, alpha=0.9)
        ax2a.add_patch(poly)

ax2a.set_xlim(*JATIM_LON_RANGE)
ax2a.set_ylim(*JATIM_LAT_RANGE)
ax2a.set_aspect('equal')
ax2a.axis('off')
ax2a.set_title('Klasifikasi Dominan (berdasarkan Prediksi)', fontsize=12, fontweight='bold')

legend_elements = [
    Patch(facecolor='#1a9641', label='HARGA RENDAH'),
    Patch(facecolor='#f7f7f7', edgecolor='#999999', label='HARGA RATA-RATA'),
    Patch(facecolor='#d73027', label='HARGA TINGGI'),
    Patch(facecolor='#e0e0e0', label='Tidak ada data'),
]
ax2a.legend(handles=legend_elements, loc='lower left', fontsize=7, framealpha=0.8)

# Panel kanan: keyakinan
for feat in jatim_features:
    alt_name = normalise_nama(feat['properties']['alt_name'])
    val = confidence_dict.get(alt_name, None)
    if val is not None:
        color = plt.cm.YlOrRd(val)
    else:
        color = '#e0e0e0'
    for coords in extract_polygon_coords(feat['geometry']):
        poly = MplPolygon(coords, closed=True, facecolor=color,
                          edgecolor='white', linewidth=0.4, alpha=0.9)
        ax2b.add_patch(poly)

ax2b.set_xlim(*JATIM_LON_RANGE)
ax2b.set_ylim(*JATIM_LAT_RANGE)
ax2b.set_aspect('equal')
ax2b.axis('off')
ax2b.set_title('Keyakinan Klasifikasi Dominan', fontsize=12, fontweight='bold')

sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=Normalize(0, 1))
sm.set_array([])
cbar2 = plt.colorbar(sm, ax=ax2b, fraction=0.04, pad=0.02, shrink=0.5)
cbar2.set_label('Proporsi waktu', fontsize=9)

fig2.suptitle(f'Pola Spasial Harga Tinggi / Harga Rendah - {KOMODITAS}\n'
              f'Sentra produksi diprediksi HARGA RENDAH vs '
              f'kab konsumsi diprediksi HARGA TINGGI',
              fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig2.savefig(os.path.join(OUTPUT_DIR, '02_konsistensi_spasial.png'), dpi=200, bbox_inches='tight')
plt.close(fig2)
print("     OK -> 02_konsistensi_spasial.png")


# ============================================================
# FIGURE 3: PETA KETIMPANGAN HARGA PER HORIZON
# ============================================================
print("\n[9] Figure 3: Peta ketimpangan harga per horizon...")

# Rata-rata deviasi harga per wilayah per horizon
deviasi_per_wilayah = (
    df_pred.groupby(['alt_name', 'horizon'])['pct_vs_provinsi_pred']
    .mean()
    .mul(100).round(2)
    .reset_index()
    .rename(columns={'pct_vs_provinsi_pred': 'deviasi_pct'})
)

horizons = sorted(deviasi_per_wilayah['horizon'].unique())
n_h = len(horizons)
n_cols, n_rows = 4, int(np.ceil(n_h / 4))

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes3 = axes3.flatten()

v_abs = deviasi_per_wilayah['deviasi_pct'].abs().quantile(0.95)
cmap_dev = LinearSegmentedColormap.from_list('dev_cmap',
    ['#1a9641', '#f7f7f7', '#d73027'], N=256)
norm_dev = Normalize(-v_abs, v_abs)

for idx, h in enumerate(horizons):
    ax = axes3[idx]
    h_dev = deviasi_per_wilayah[deviasi_per_wilayah['horizon'] == h]
    dev_dict = dict(zip(h_dev['alt_name'], h_dev['deviasi_pct']))

    for feat in jatim_features:
        alt_name = normalise_nama(feat['properties']['alt_name'])
        val = dev_dict.get(alt_name, None)
        if val is not None:
            color = cmap_dev(norm_dev(val))
        else:
            color = '#e0e0e0'
        for coords in extract_polygon_coords(feat['geometry']):
            poly = MplPolygon(coords, closed=True, facecolor=color,
                              edgecolor='white', linewidth=0.3, alpha=0.9)
            ax.add_patch(poly)

    ax.set_xlim(*JATIM_LON_RANGE)
    ax.set_ylim(*JATIM_LAT_RANGE)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'H+{h}', fontsize=12, fontweight='bold')

for idx in range(n_h, len(axes3)):
    axes3[idx].axis('off')

sm3 = plt.cm.ScalarMappable(cmap=cmap_dev, norm=norm_dev)
sm3.set_array([])
cbar3 = fig3.colorbar(sm3, ax=axes3, orientation='horizontal',
                      fraction=0.02, pad=0.02, aspect=50)
cbar3.set_label('Deviasi harga dari rata-rata provinsi (%) - '
                'Hijau = HARGA RENDAH, Merah = HARGA TINGGI', fontsize=10)

fig3.suptitle(f'Ketimpangan Harga {KOMODITAS} per Horizon Prediksi\n'
              f'Rata-rata deviasi harga tiap Kabupaten/Kota terhadap rata-rata provinsi - Periode Uji 2024-2025',
              fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig(os.path.join(OUTPUT_DIR, '03_ketimpangan_per_horizon.png'), dpi=200, bbox_inches='tight')
plt.close(fig3)
print("     OK -> 03_ketimpangan_per_horizon.png")


# ============================================================
# FIGURE 4: ANALISIS TEMPORAL - KETIMPANGAN vs HARI LIBUR
# ============================================================
print("\n[10] Figure 4: Analisis temporal ketimpangan vs hari libur...")

# CoV harian
daily_cov = (
    df_pred.groupby('forecast_date')['prediction']
    .agg(['mean', 'std'])
    .reset_index()
)
daily_cov['cov'] = daily_cov['std'] / daily_cov['mean'].clip(lower=1)
daily_cov['cov_pct'] = daily_cov['cov'] * 100

# CoV harian (actual)
daily_cov_actual = (
    df_pred.groupby('forecast_date')['actual']
    .agg(['mean', 'std'])
    .reset_index()
)
daily_cov_actual['cov'] = daily_cov_actual['std'] / daily_cov_actual['mean'].clip(lower=1)
daily_cov_actual['cov_pct_actual'] = daily_cov_actual['cov'] * 100

# Merge
daily_cov = daily_cov.merge(daily_cov_actual[['forecast_date', 'cov_pct_actual']],
                            on='forecast_date', how='left')

# Periode libur
holiday_periods = get_holiday_periods(df_libur)

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]})

# --- Panel Atas: CoV harian dengan highlight libur ---
ax4a.plot(daily_cov['forecast_date'], daily_cov['cov_pct'],
          color='#2c3e50', linewidth=0.7, alpha=0.8, label='CoV prediksi')
ax4a.plot(daily_cov['forecast_date'], daily_cov['cov_pct_actual'],
          color='#e67e22', linewidth=0.7, alpha=0.5, label='CoV aktual')

y_min = min(daily_cov['cov_pct'].min(), daily_cov['cov_pct_actual'].min())
y_max = max(daily_cov['cov_pct'].max(), daily_cov['cov_pct_actual'].max())
y_pad = (y_max - y_min) * 0.05
legend_handles = []

for start, end, label, color, alpha in holiday_periods:
    if end < daily_cov['forecast_date'].min() or start > daily_cov['forecast_date'].max():
        continue
    ax4a.axvspan(max(start, daily_cov['forecast_date'].min()),
                 min(end, daily_cov['forecast_date'].max()),
                 color=color, alpha=alpha, zorder=0)
    lbl = 'Lebaran' if 'Lebaran' in label else 'Nataru'
    if lbl not in [h.get_label() for h in legend_handles]:
        legend_handles.append(Patch(facecolor=color, alpha=alpha, label=lbl))

# Rata-rata CoV periode libur vs non-libur
holiday_mask = pd.Series(False, index=daily_cov.index)
for start, end, *_ in holiday_periods:
    holiday_mask |= (daily_cov['forecast_date'] >= start) & \
                    (daily_cov['forecast_date'] <= end)

cov_holiday = daily_cov.loc[holiday_mask, 'cov_pct'].mean()
cov_nonholiday = daily_cov.loc[~holiday_mask, 'cov_pct'].mean()
cov_holiday_act = daily_cov.loc[holiday_mask, 'cov_pct_actual'].mean()
cov_nonholiday_act = daily_cov.loc[~holiday_mask, 'cov_pct_actual'].mean()

ax4a.axhline(cov_holiday, color='#e74c3c', linewidth=1.5, linestyle='--',
             label=f'Rerata libur (prediksi: {cov_holiday:.2f}%)')
ax4a.axhline(cov_nonholiday, color='#3498db', linewidth=1.5, linestyle='--',
             label=f'Rerata non-libur (prediksi: {cov_nonholiday:.2f}%)')

delta_cov = cov_holiday - cov_nonholiday
delta_cov_act = cov_holiday_act - cov_nonholiday_act
ax4a.annotate(f'Delta CoV (prediksi): {delta_cov:+.2f} pp\nDelta CoV (aktual): {delta_cov_act:+.2f} pp',
              xy=(0.02, 0.95), xycoords='axes fraction', fontsize=11, fontweight='bold',
              color='#c0392b' if delta_cov > 0 else '#27ae60',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax4a.set_ylabel('CoV (%) - Coefficient of Variation', fontsize=11, fontweight='bold')
ax4a.set_title(f'Ketimpangan Harga {KOMODITAS} Antar Wilayah dari Waktu ke Waktu\n'
               f'CoV harian (std/mean) - makin tinggi = makin timpang',
               fontsize=13, fontweight='bold', pad=10)
ax4a.legend(loc='upper right', fontsize=8, framealpha=0.8)
ax4a.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax4a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax4a.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax4a.set_ylim(y_min - y_pad, y_max + y_pad * 2)

# --- Panel Bawah: Perbandingan klasifikasi prediksi vs aktual ---
daily_agree = (
    df_pred.groupby('forecast_date')
    .apply(lambda g: (g['klasifikasi_pred'] == g['klasifikasi_actual']).mean())
    .reset_index()
    .rename(columns={0: 'agreement'})
)

ax4b.plot(daily_agree['forecast_date'], daily_agree['agreement'] * 100,
          color='#8e44ad', linewidth=0.7, alpha=0.8)
ax4b.axhline(daily_agree['agreement'].mean() * 100, color='#8e44ad',
             linewidth=1.5, linestyle='--',
             label=f'Rerata: {daily_agree["agreement"].mean()*100:.1f}%')

# Highlight libur juga
for start, end, label, color, alpha in holiday_periods:
    if end < daily_cov['forecast_date'].min() or start > daily_cov['forecast_date'].max():
        continue
    ax4b.axvspan(max(start, daily_cov['forecast_date'].min()),
                 min(end, daily_cov['forecast_date'].max()),
                 color=color, alpha=alpha * 0.6, zorder=0)

ax4b.set_xlabel('Tanggal', fontsize=12, fontweight='bold')
ax4b.set_ylabel('Agreement Prediksi vs Aktual (%)', fontsize=11, fontweight='bold')
ax4b.set_title('Seberapa sering klasifikasi dari PREDIKSI cocok dengan AKTUAL?', fontsize=12, fontweight='bold')
ax4b.legend(loc='lower left', fontsize=9)
ax4b.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax4b.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax4b.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax4b.set_ylim(0, 100)

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, '04_temporal_hari_libur.png'), dpi=200, bbox_inches='tight')
plt.close(fig4)
print("     OK -> 04_temporal_hari_libur.png")


# ============================================================
# RINGKASAN
# ============================================================
print("\n" + "=" * 60)
print("  RINGKASAN HASIL ANALISIS")
print("=" * 60)

for src, label in [('pred', 'Prediksi'), ('actual', 'Aktual')]:
    counts = df_pred[f'klasifikasi_{src}'].value_counts()
    total = counts.sum()
    print(f"\n  Distribusi klasifikasi ({label}):")
    for cat in ['Harga Rendah', 'Harga Rata-rata', 'Harga Tinggi']:
        n = counts.get(cat, 0)
        print(f"     {cat:10s} : {n:>8,} ({n / total:.1%})")

print(f"\n  Rata-rata CoV: {daily_cov['cov_pct'].mean():.2f}%")
print(f"     Periode libur     : {cov_holiday:.2f}%")
print(f"     Periode non-libur : {cov_nonholiday:.2f}%")
print(f"     Delta             : {delta_cov:+.2f} pp")
print(f"\n  Agreement prediksi vs aktual: {agreement:.1%}")

harga_rendah_wilayah = dominant_class[dominant_class['klasifikasi'] == 'Harga Rendah'].nlargest(3, 'persentase')
harga_tinggi_wilayah = dominant_class[dominant_class['klasifikasi'] == 'Harga Tinggi'].nlargest(3, 'persentase')
print(f"\n  Top 3 wilayah Harga Rendah (potensi sumber distribusi):")
for _, r in harga_rendah_wilayah.iterrows():
    print(f"     {r['alt_name'].title():30s} {r['persentase']:.1f}%")
print(f"\n  Top 3 wilayah Harga Tinggi (prioritas tujuan distribusi):")
for _, r in harga_tinggi_wilayah.iterrows():
    print(f"     {r['alt_name'].title():30s} {r['persentase']:.1f}%")

print(f"\n  Visualisasi tersimpan di: {OUTPUT_DIR}/")
print("  Selesai.")