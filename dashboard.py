"""
Dashboard Prediksi Harga & Rekomendasi Distribusi Pangan Jawa Timur
==================================================================================
Kebutuhan:  pip install streamlit pandas numpy plotly networkx

Jalankan:   streamlit run dashboard.py
Catatan:    Taruh dashboard.py di folder yang sama dengan file data (CSV & GeoJSON).
            Dashboard ini menggunakan hasil prediksi 2024-2025 (pre-computed),
            bukan arsitektur realtime.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Distribusi Pangan Jawa Timur",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── TEMA & CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* =====================================================================
   TOKEN WARNA — ubah di sini saja untuk retheming
   ===================================================================== */
:root {
    --bg-page:        #f8f9fb;
    --bg-card:        #ffffff;
    --bg-sidebar:     #ffffff;
    --border:         #e4e7ec;
    --border-strong:  #d0d5dd;
    --text-primary:   #101828;
    --text-secondary: #475467;
    --text-muted:     #98a2b3;
    --accent-blue:    #1d6ae5;
    --accent-green:   #12b76a;
    --accent-amber:   #f79009;
    --accent-red:     #f04438;
    --accent-purple:  #7f56d9;
    --badge-low-bg:   #ecfdf3;
    --badge-low-text: #027a48;
    --badge-avg-bg:   #fffaeb;
    --badge-avg-text: #b54708;
    --badge-hi-bg:    #fef3f2;
    --badge-hi-text:  #b42318;
}

/* =====================================================================
   1. GLOBAL
   ===================================================================== */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 14px;
}
.stApp {
    background-color: var(--bg-page);
    color: var(--text-primary);
}
h1, h2, h3, h4 { color: var(--text-primary) !important; font-weight: 700; }
p, li           { color: var(--text-secondary); }

/* =====================================================================
   2. SIDEBAR
   ===================================================================== */
section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

/* =====================================================================
   3. HEADER STREAMLIT
   ===================================================================== */
header[data-testid="stHeader"] {
    background-color: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
}
header[data-testid="stHeader"] svg { fill: var(--text-primary) !important; }

/* =====================================================================
   4. INPUT KOMPONEN
   ===================================================================== */
div[data-baseweb="select"],
div[data-baseweb="input"],
div[data-baseweb="select"] > div,
div[data-baseweb="input"] input {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
div[data-baseweb="select"] *,
div[data-baseweb="input"] * { color: var(--text-primary) !important; }
div[data-baseweb="select"] svg { fill: var(--text-secondary) !important; }

.stSelectbox label,
.stDateInput label,
.stSlider label {
    color: var(--text-secondary) !important;
    font-weight: 500;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* =====================================================================
   5. POPUP / KALENDER / DROPDOWN
   ===================================================================== */
div[role="dialog"],
div[role="listbox"],
div[data-baseweb="popover"],
div[data-baseweb="calendar"],
div[data-baseweb="datepicker"],
div[data-baseweb="datepicker-year-select"],
div[data-baseweb="datepicker-month-select"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 8px 24px rgba(16,24,40,0.08) !important;
}

/* Semua teks di dalam kalender */
div[data-baseweb="calendar"],
div[data-baseweb="calendar"] *,
div[data-baseweb="calendar"] span,
div[data-baseweb="calendar"] div,
div[data-baseweb="calendar"] td,
div[data-baseweb="datepicker"] * {
    color: var(--text-primary) !important;
    background-color: #ffffff !important;
}
.st-ge::after {
    background-color: rgb(255 255 255);
}

/* Tombol tanggal aktif */
div[data-baseweb="calendar"] button {
    background-color: transparent !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}
div[data-baseweb="calendar"] button:hover {
    background-color: #f0f4ff !important;
}

/* Tanggal yang dipilih */
div[data-baseweb="calendar"] [aria-selected="true"],
div[data-baseweb="calendar"] [aria-selected="true"] * {
    background-color: var(--accent-blue) !important;
    color: #ffffff !important;
    border-radius: 6px !important;
}

/* Tanggal disabled (di luar rentang/bulan) — paksa warna abu-abu */
div[data-baseweb="calendar"] button[disabled],
div[data-baseweb="calendar"] [aria-disabled="true"],
div[data-baseweb="calendar"] button[tabindex="-1"] {
    color: var(--text-muted) !important;
    opacity: 0.35 !important;
}

/* Label hari (Minggu-Sabtu) */
div[data-baseweb="calendar"] thead th,
div[data-baseweb="calendar"] thead span,
div[data-baseweb="calendar"] thead * {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    background: transparent !important;
}

/* Header bulan/tahun */
div[data-baseweb="calendar"] select,
div[data-baseweb="calendar"] select option,
div[data-baseweb="calendar"] div[role="heading"],
div[data-baseweb="calendar"] div[role="heading"] *,
div[data-baseweb="calendar"] button[aria-label*="Switch"],
div[data-baseweb="calendar"] button[aria-label*="Move"],
div[data-baseweb="calendar"] button[aria-label*="Next"],
div[data-baseweb="calendar"] button[aria-label*="Previous"] {
    color: var(--text-primary) !important;
    background-color: transparent !important;
}

/* =====================================================================
   7. TAB NAVIGASI
   ===================================================================== */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-page);
    gap: 2px;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    border-radius: 7px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background-color: rgba(0,0,0,0.03);
}
.stTabs [aria-selected="true"] {
    background-color: var(--bg-card) !important;
    color: var(--accent-blue) !important;
    font-weight: 600;
    box-shadow: 0 1px 4px rgba(16,24,40,0.10);
}

/* =====================================================================
   8. METRIK
   ===================================================================== */
div[data-testid="metric-container"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* =====================================================================
   9. BADGE & CHIP STATUS
   ===================================================================== */
.badge-low {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--badge-low-bg); color: var(--badge-low-text);
    font-weight: 600; font-size: 12px;
    padding: 3px 10px; border-radius: 20px;
    border: 1px solid #abefc6;
}
.badge-avg {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--badge-avg-bg); color: var(--badge-avg-text);
    font-weight: 600; font-size: 12px;
    padding: 3px 10px; border-radius: 20px;
    border: 1px solid #fedf89;
}
.badge-high {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--badge-hi-bg); color: var(--badge-hi-text);
    font-weight: 600; font-size: 12px;
    padding: 3px 10px; border-radius: 20px;
    border: 1px solid #fecdca;
}

/* =====================================================================
   10. INFO / WARNING / ERROR / SUCCESS BOXES
   ===================================================================== */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-width: 1px !important;
    font-size: 13px !important;
}
/* Info — biru muda */
div[data-testid="stAlert"]:has(div[data-baseweb="notification"][kind="info"]) {
    background-color: #f0f9ff !important;
    border-color: #bae6fd !important;
    color: #0c4a6e !important;
}
/* Warning — kuning muda */
div[data-testid="stAlert"]:has(div[data-baseweb="notification"][kind="warning"]) {
    background-color: #fffaeb !important;
    border-color: #fedf89 !important;
    color: #b54708 !important;
}
/* Error — merah muda */
div[data-testid="stAlert"]:has(div[data-baseweb="notification"][kind="error"]) {
    background-color: #fef3f2 !important;
    border-color: #fecdca !important;
    color: #b42318 !important;
}
/* Success — hijau muda */
div[data-testid="stAlert"]:has(div[data-baseweb="notification"][kind="success"]) {
    background-color: #ecfdf3 !important;
    border-color: #abefc6 !important;
    color: #027a48 !important;
}
/* Teks di dalam alert */
div[data-testid="stAlert"] div[data-baseweb="notification"] span,
div[data-testid="stAlert"] div[data-baseweb="notification"] p,
div[data-testid="stAlert"] div[data-baseweb="notification"] a {
    color: inherit !important;
}
/* Svg icon di alert */
div[data-testid="stAlert"] svg {
    fill: currentColor !important;
    opacity: 0.7;
}

/* Spinner — konsisten tema terang */
div[data-testid="stSpinner"] {
    background-color: transparent !important;
}
div[data-testid="stSpinner"] > div {
    border-top-color: var(--accent-blue) !important;
    border-right-color: var(--accent-blue) !important;
    border-bottom-color: var(--accent-blue) !important;
    border-left-color: var(--border-strong) !important;
}
div[data-testid="stSpinner"] + div {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* =====================================================================
   11. CARD KOMPONEN KUSTOM
   ===================================================================== */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.stat-card .label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.stat-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}
.stat-card .delta {
    font-size: 11.5px;
    font-weight: 500;
    margin-top: 2px;
}
.delta-pos { color: #027a48; }
.delta-neg { color: #b42318; }

/* Sidebar section header */
.sidebar-section {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 16px 0 6px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
}

/* =====================================================================
   12. FIX DARK-MODE WIDGET OVERRIDE
   ===================================================================== */

/* Paksa semua BaseUI component ke light */
[data-baseweb] {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* Dropdown menu items */
ul[role="listbox"],
ul[data-baseweb="menu"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 16px rgba(16,24,40,0.08) !important;
}
li[role="option"],
ul[data-baseweb="menu"] li {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}
li[role="option"]:hover,
ul[data-baseweb="menu"] li:hover {
    background-color: #f0f4ff !important;
}

/* Slider track & thumb */
div[data-testid="stSlider"] > div {
    background: transparent !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent-blue) !important;
}

/* Select input value text */
div[data-baseweb="select"] input,
div[data-baseweb="select"] [data-id="selected-option"],
div[data-baseweb="select"] span {
    color: var(--text-primary) !important;
    background-color: transparent !important;
}

/* Streamlit dataframe iframe fix */
.stDataFrame iframe,
.stDataFrame [data-testid="stDataFrameResizable"] {
    color-scheme: light !important;
}

/* Page section header */
.section-header {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.section-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 16px;
}

/* =====================================================================
   13. CLEANUP BAWAAN
   ===================================================================== */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── KONSTANTA ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
THRESHOLD_KM = 120

# Warna untuk peta & scatter
STATUS_COLOR = {
    "Harga Rendah":    "#12b76a",
    "Harga Rata-rata": "#f79009",
    "Harga Tinggi":    "#f04438",
    "Tidak Ada Data":  "#98a2b3",
}

# Warna Plotly yang konsisten dengan tema terang
C_BLUE   = "#1d6ae5"
C_GREEN  = "#12b76a"
C_AMBER  = "#f79009"
C_RED    = "#f04438"
C_PURPLE = "#7f56d9"
C_GRAY   = "#98a2b3"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", color="#475467", size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(
        orientation="h", y=1.08,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#475467"),
    ),
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#e4e7ec",
        font=dict(family="Inter, sans-serif", size=12, color="#101828"),
    ),
    xaxis=dict(
        gridcolor="#f0f1f3",
        linecolor="#e4e7ec",
        tickfont=dict(size=11, color="#667085"),
        title_font=dict(size=11, color="#667085"),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#f0f1f3",
        linecolor="#e4e7ec",
        tickfont=dict(size=11, color="#667085"),
        title_font=dict(size=11, color="#667085"),
        zeroline=False,
    ),
)

HOLIDAY_PERIODS = [
    ("2024-03-28", "2024-04-16", "Lebaran 2024"),
    ("2024-12-22", "2025-01-05", "Nataru 2024/25"),
    ("2025-03-17", "2025-04-08", "Lebaran 2025"),
    ("2025-12-22", "2025-12-31", "Nataru 2025/26"),
]

# ── UTILITAS ────────────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    n = str(name).upper().strip()
    for prefix in ("KABUPATEN ", "KOTA "):
        n = n.replace(prefix, "")
    return n


def short_name(name: str) -> str:
    return name.replace("Kabupaten ", "Kab. ").replace("Kota ", "")


def apply_plotly_theme(fig: go.Figure, height: int = 360, **overrides) -> go.Figure:
    """Terapkan layout standar tema terang ke semua figure Plotly.
       Override tickfont y-axis dengan warna lebih gelap untuk axis labels."""
    layout = {**PLOTLY_LAYOUT, "height": height, **overrides}
    # Pastikan tickfont y-axis cukup gelap
    if "yaxis" in overrides and overrides["yaxis"].get("tickfont") is None:
        pass  # biarkan custom override yaxis bekerja
    elif "yaxis" not in overrides:
        # Default y-axis dengan warna lebih gelap
        layout["yaxis"] = {
            **PLOTLY_LAYOUT["yaxis"],
            "tickfont": dict(size=11, color="#344054"),  # lebih gelap dari #667085
        }
    fig.update_layout(**layout)
    return fig


# ── FUNGSI LOAD DATA ────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions(komoditas: str) -> pd.DataFrame:
    fname = (
        "test_predictions_bawang_merah.csv"
        if komoditas == "Bawang Merah"
        else "test_predictions_cabai_rawit.csv"
    )
    df = pd.read_csv(BASE / fname, parse_dates=["forecast_date"])
    return df


@st.cache_data
def load_coords() -> pd.DataFrame:
    return pd.read_csv(BASE / "koordinat_jatim.csv")


@st.cache_data
def load_osrm() -> pd.DataFrame:
    return pd.read_csv(BASE / "jarak_waktu_tempuh_osrm.csv")


@st.cache_data
def load_geojson() -> dict:
    with open(BASE / "all_kabkota_ind.geojson", encoding="utf-8") as f:
        gj = json.load(f)
    jatim = [
        feat for feat in gj["features"]
        if feat["properties"].get("province_id") == "35"
    ]
    for feat in jatim:
        norm = normalize_name(feat["properties"]["name"])
        feat["id"] = norm
        feat["properties"]["norm_name"] = norm
    return {"type": "FeatureCollection", "features": jatim}


# ── FUNGSI KALKULASI ─────────────────────────────────────────────────────────────
@st.cache_data
def compute_classification(df: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for (date, h), grp in df.groupby(["forecast_date", "horizon"]):
        g = grp.copy()
        p25 = g["prediction"].quantile(0.25)
        p75 = g["prediction"].quantile(0.75)
        prov_mean = g["prediction"].mean()
        g["status"] = g["prediction"].apply(
            lambda x: "Harga Rendah" if x <= p25 else
                      ("Harga Tinggi" if x >= p75 else "Harga Rata-rata")
        )
        g["prov_mean"] = prov_mean
        g["pct_vs_prov"] = (g["prediction"] - prov_mean) / prov_mean * 100
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True)


@st.cache_data
def compute_cov(df_cls: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_cls
        .groupby(["forecast_date", "horizon"])
        .agg(
            pred_mean=("prediction", "mean"),
            pred_std=("prediction", "std"),
            actual_mean=("actual", "mean"),
            actual_std=("actual", "std"),
        )
        .reset_index()
    )
    agg["cov_pred"]   = agg["pred_std"]   / agg["pred_mean"]   * 100
    agg["cov_actual"] = agg["actual_std"] / agg["actual_mean"] * 100
    return agg


@st.cache_data
def compute_centrality(threshold_km: float = THRESHOLD_KM):
    osrm = load_osrm()
    edges = osrm[osrm["jarak_km"] <= threshold_km]
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["kab_asal"], row["kab_tujuan"], weight=row["jarak_km"])
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
    deg_df = (
        pd.DataFrame(deg.items(), columns=["kab_kota", "degree"])
        .sort_values("degree", ascending=False)
        .reset_index(drop=True)
    )
    btw_df = (
        pd.DataFrame(btw.items(), columns=["kab_kota", "betweenness"])
        .sort_values("betweenness", ascending=False)
        .reset_index(drop=True)
    )
    return deg_df, btw_df, G


def get_recommendations(
    df_cls: pd.DataFrame,
    date,
    horizon: int,
    osrm: pd.DataFrame,
    threshold_km: float = THRESHOLD_KM,
    top_n: int = 25,
) -> pd.DataFrame:
    snapshot = df_cls[
        (df_cls["forecast_date"] == pd.Timestamp(date)) &
        (df_cls["horizon"] == horizon)
    ]
    if snapshot.empty:
        return pd.DataFrame()

    sources = snapshot[snapshot["status"] == "Harga Rendah"][["kab_kota", "prediction"]]
    targets = snapshot[snapshot["status"] == "Harga Tinggi"][["kab_kota", "prediction"]]
    if sources.empty or targets.empty:
        return pd.DataFrame()

    direct = osrm[osrm["jarak_km"] <= threshold_km]
    rows = []
    for _, src in sources.iterrows():
        for _, tgt in targets.iterrows():
            edge = direct[
                ((direct["kab_asal"] == src["kab_kota"]) & (direct["kab_tujuan"] == tgt["kab_kota"])) |
                ((direct["kab_asal"] == tgt["kab_kota"]) & (direct["kab_tujuan"] == src["kab_kota"]))
            ]
            if edge.empty:
                continue
            e = edge.iloc[0]
            selisih = tgt["prediction"] - src["prediction"]
            rows.append({
                "Sumber":        src["kab_kota"],
                "Tujuan":        tgt["kab_kota"],
                "Harga Sumber":  int(src["prediction"]),
                "Harga Tujuan":  int(tgt["prediction"]),
                "Selisih Harga": int(selisih),
                "Jarak (km)":    round(e["jarak_km"], 1),
                "Waktu (menit)": round(e["waktu_menit"], 0),
            })

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("Selisih Harga", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── LOAD DATA AWAL ──────────────────────────────────────────────────────────────
coords  = load_coords()
osrm    = load_osrm()
geojson = load_geojson()

coord_dict: dict[str, tuple[float, float]] = {
    row["nama"]: (row["longitude"], row["latitude"])
    for _, row in coords.iterrows()
}

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown("""
    <div style="padding: 20px 16px 16px 16px; border-bottom: 1px solid #e4e7ec; margin-bottom: 4px;">
        <div style="font-size: 14px; font-weight: 700; color: #101828;">
            Distribusi Pangan
        </div>
        <div style="font-size: 11px; color: #98a2b3; margin-top: 2px;">
            Jawa Timur · 38 Kab/Kota
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">Filter Data</div>
    """, unsafe_allow_html=True)

    komoditas = st.selectbox(
        "Komoditas",
        ["Cabai Rawit Merah", "Bawang Merah"],
    )

    # Load & hitung
    df_raw = load_predictions(komoditas)
    df_cls = compute_classification(df_raw)
    df_cov = compute_cov(df_cls)

    available_dates = sorted(df_raw["forecast_date"].dt.date.unique())
    min_d, max_d    = available_dates[0], available_dates[-1]

    selected_date = st.date_input(
        "Tanggal Prediksi",
        value=min_d,
        min_value=min_d,
        max_value=max_d,
    )

    horizon = st.select_slider(
        "Horizon Prediksi",
        options=[1, 2, 3, 4, 5, 6, 7],
        value=1,
        format_func=lambda x: f"H+{x}",
    )

    # Snapshot untuk sidebar stats
    snapshot = df_cls[
        (df_cls["forecast_date"] == pd.Timestamp(selected_date)) &
        (df_cls["horizon"] == horizon)
    ]

    st.markdown("""
    <div class="sidebar-section" style="margin-top: 20px;">Status Wilayah</div>
    """, unsafe_allow_html=True)

    if not snapshot.empty:
        n_high    = int((snapshot["status"] == "Harga Tinggi").sum())
        n_low     = int((snapshot["status"] == "Harga Rendah").sum())
        n_avg     = int((snapshot["status"] == "Harga Rata-rata").sum())
        prov_mean = snapshot["prediction"].mean()

        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-bottom: 10px;">
            <div style="background: #fef3f2; border: 1px solid #fecdca; border-radius: 8px;
                        padding: 10px 8px; text-align: center;">
                <div style="font-size: 18px; font-weight: 700; color: #b42318;">{n_high}</div>
                <div style="font-size: 10px; color: #b42318; font-weight: 600;">TINGGI</div>
            </div>
            <div style="background: #ecfdf3; border: 1px solid #abefc6; border-radius: 8px;
                        padding: 10px 8px; text-align: center;">
                <div style="font-size: 18px; font-weight: 700; color: #027a48;">{n_low}</div>
                <div style="font-size: 10px; color: #027a48; font-weight: 600;">RENDAH</div>
            </div>
            <div style="background: #fffaeb; border: 1px solid #fedf89; border-radius: 8px;
                        padding: 10px 8px; text-align: center;">
                <div style="font-size: 18px; font-weight: 700; color: #b54708;">{n_avg}</div>
                <div style="font-size: 10px; color: #b54708; font-weight: 600;">NORMAL</div>
            </div>
        </div>
        <div style="background: #f8f9fb; border: 1px solid #e4e7ec; border-radius: 8px;
                    padding: 10px 12px; margin-bottom: 6px;">
            <div style="font-size: 10px; font-weight: 600; color: #98a2b3;
                        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px;">
                Rata-rata Provinsi
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #101828;">
                Rp {prov_mean:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section" style="margin-top: 16px;">Tentang Model</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 11.5px; color: #667085; line-height: 1.7; padding: 4px 0;">
        <b style="color:#475467">Model</b>: LightGBM + Recursive Multi-Step<br>
        <b style="color:#475467">Data</b>: SISKAPERBAPO · Open-Meteo · OSRM<br>
        <b style="color:#475467">Periode uji</b>: Jan 2024 – Des 2025
    </div>
    <div style="font-size: 10.5px; color: #98a2b3; margin-top: 8px; font-style: italic;">
        Arsitektur realtime: pengembangan lanjutan.
    </div>
    """, unsafe_allow_html=True)


# ── HEADER HALAMAN ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: #ffffff; border: 1px solid #e4e7ec; border-radius: 12px;
            padding: 20px 24px; margin-bottom: 20px;
            display: flex; align-items: center; justify-content: space-between;">
    <div>
        <h1 style="font-size: 1.4rem; font-weight: 800; margin: 0; color: #101828 !important;">
            Sistem Prediksi Harga & Rekomendasi Distribusi Pangan Jawa Timur
        </h1>
        <p style="color: #667085; margin: 4px 0 0 0; font-size: 13px;">
            {komoditas} &nbsp;·&nbsp; 38 Kabupaten/Kota &nbsp;·&nbsp;
            Tampilan: <strong>H+{horizon}</strong> &nbsp;·&nbsp; {selected_date}
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Peta Harga",
    "Prediksi per Wilayah",
    "Rekomendasi Distribusi",
    "Analisis Ketimpangan",
    "Sentralitas Graf",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PETA HARGA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div class="section-header">Peta Status Harga</div>
    <div class="section-sub">{komoditas} · H+{horizon} · {selected_date}</div>
    """, unsafe_allow_html=True)

    if snapshot.empty:
        st.warning("Tidak ada data untuk tanggal ini. Coba pilih tanggal lain.")
    else:
        map_df = snapshot.copy()
        map_df["norm_name"]    = map_df["kab_kota"].apply(normalize_name)
        map_df["color_num"]    = map_df["status"].map(
            {"Harga Rendah": 0, "Harga Rata-rata": 1, "Harga Tinggi": 2}
        )
        map_df["Prediksi (Rp)"] = map_df["prediction"].map(lambda x: f"Rp {x:,.0f}")
        map_df["Aktual (Rp)"]   = map_df["actual"].map(lambda x: f"Rp {x:,.0f}")
        map_df["Deviasi (%)"]   = map_df["pct_vs_prov"].map(lambda x: f"{x:+.1f}%")

        # Skala warna terang: hijau → kuning → merah
        fig_map = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="norm_name",
            featureidkey="id",
            color="color_num",
            color_continuous_scale=[
                [0.00, "#d1fae5"],
                [0.25, "#34d399"],
                [0.50, "#fde68a"],
                [0.75, "#fca5a5"],
                [1.00, "#ef4444"],
            ],
            range_color=[0, 2],
            mapbox_style="carto-positron",
            center={"lat": -7.54, "lon": 112.20},
            zoom=7.0,
            opacity=0.82,
            hover_name="kab_kota",
            hover_data={
                "Prediksi (Rp)": True,
                "Aktual (Rp)":   True,
                "status":        True,
                "Deviasi (%)":   True,
                "norm_name":     False,
                "color_num":     False,
            },
            labels={"status": "Status"},
            height=500,
        )
        fig_map.update_layout(
            paper_bgcolor="#ffffff",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Legenda badge
        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown('<span class="badge-low">● Harga Rendah — sumber distribusi</span>', unsafe_allow_html=True)
        lc2.markdown('<span class="badge-avg">● Harga Rata-rata — kondisi normal</span>',  unsafe_allow_html=True)
        lc3.markdown('<span class="badge-high">● Harga Tinggi — prioritas tujuan</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabel ringkasan
        st.markdown("""
        <div class="section-header" style="font-size:13px;">Ringkasan Status 38 Kabupaten/Kota</div>
        """, unsafe_allow_html=True)
        disp = (
            map_df[["kab_kota", "Prediksi (Rp)", "Aktual (Rp)", "status", "Deviasi (%)"]]
            .rename(columns={"kab_kota": "Kabupaten/Kota", "status": "Status"})
            .sort_values("Status")
        )
        st.dataframe(disp, use_container_width=True, hide_index=True, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI PER WILAYAH
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">Grafik Prediksi H+1–H+7 per Wilayah</div>
    <div class="section-sub">Pilih kabupaten/kota untuk melihat perbandingan prediksi vs aktual.</div>
    """, unsafe_allow_html=True)

    all_kabs = sorted(df_raw["kab_kota"].unique())
    sel_kab  = st.selectbox("Pilih Kabupaten/Kota", all_kabs, key="kab_tab2")

    kab_data = df_raw[df_raw["kab_kota"] == sel_kab]
    fan_data = kab_data[kab_data["forecast_date"] == pd.Timestamp(selected_date)]

    if fan_data.empty:
        st.info("Tidak ada data prediksi untuk kabupaten dan tanggal yang dipilih.")
    else:
        col_fan, col_ts = st.columns([1, 2])

        # — Fan chart H+1..H+7
        with col_fan:
            st.markdown(f"""
            <div style="font-size: 12px; font-weight: 600; color: #667085;
                        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                Prediksi dari {selected_date}
            </div>
            """, unsafe_allow_html=True)
            fig_fan = go.Figure()
            fig_fan.add_trace(go.Scatter(
                x=[f"H+{h}" for h in fan_data["horizon"]],
                y=fan_data["prediction"],
                mode="lines+markers",
                name="Prediksi",
                line=dict(color=C_BLUE, width=2.5),
                marker=dict(size=9, color=C_BLUE,
                            line=dict(color="#ffffff", width=2)),
            ))
            fig_fan.add_trace(go.Scatter(
                x=[f"H+{h}" for h in fan_data["horizon"]],
                y=fan_data["actual"],
                mode="lines+markers",
                name="Aktual",
                line=dict(color=C_GREEN, width=2, dash="dash"),
                marker=dict(size=9, symbol="diamond", color=C_GREEN,
                            line=dict(color="#ffffff", width=2)),
            ))
            apply_plotly_theme(fig_fan, height=300,
                               margin=dict(l=50, r=10, t=10, b=40),
                               yaxis=dict(gridcolor="#f0f1f3", tickformat=",.0f",
                                          title="Harga (Rp)",
                                          tickfont=dict(size=11, color="#667085"),
                                          title_font=dict(size=11, color="#667085")))
            st.plotly_chart(fig_fan, use_container_width=True)

        # — Time series H+1
        with col_ts:
            st.markdown("""
            <div style="font-size: 12px; font-weight: 600; color: #667085;
                        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                Tren Harga H+1 Sepanjang Periode Uji
            </div>
            """, unsafe_allow_html=True)
            h1 = kab_data[kab_data["horizon"] == 1].sort_values("forecast_date")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=h1["forecast_date"], y=h1["actual"],
                mode="lines", name="Aktual",
                line=dict(color=C_GREEN, width=1.6),
                fill="tozeroy",
                fillcolor="rgba(18,183,106,0.06)",
            ))
            fig_ts.add_trace(go.Scatter(
                x=h1["forecast_date"], y=h1["prediction"],
                mode="lines", name="Prediksi H+1",
                line=dict(color=C_BLUE, width=1.6),
            ))
            for start, _, label in HOLIDAY_PERIODS:
                fig_ts.add_vline(
                    x=start, line_width=1,
                    line_dash="dot", line_color=C_AMBER,
                )
                fig_ts.add_annotation(
                    x=start, y=1.0, yref="paper",
                    text=label, showarrow=False,
                    font=dict(color=C_AMBER, size=8),
                    xanchor="left", yanchor="bottom",
                )
            apply_plotly_theme(fig_ts, height=300,
                               margin=dict(l=50, r=10, t=10, b=40),
                               yaxis=dict(gridcolor="#f0f1f3", tickformat=",.0f",
                                          title="Harga (Rp)",
                                          tickfont=dict(size=11, color="#667085"),
                                          title_font=dict(size=11, color="#667085")))
            st.plotly_chart(fig_ts, use_container_width=True)

    # Tabel akurasi
    st.markdown("""<div style="border-top: 1px solid #e4e7ec; margin: 16px 0;"></div>""",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header" style="font-size:13px;">Metrik Akurasi Model per Horizon</div>
    """, unsafe_allow_html=True)
    met_rows = []
    for h in range(1, 8):
        hd = kab_data[kab_data["horizon"] == h].dropna(subset=["actual", "prediction"])
        if len(hd) < 2:
            continue
        err  = hd["prediction"] - hd["actual"]
        mae  = np.abs(err).mean()
        rmse = np.sqrt((err ** 2).mean())
        mape = (np.abs(err / hd["actual"]) * 100).mean()
        ss_res = (err ** 2).sum()
        ss_tot = ((hd["actual"] - hd["actual"].mean()) ** 2).sum()
        r2 = max(1 - ss_res / ss_tot if ss_tot > 0 else 0, 0)
        met_rows.append({
            "Horizon":   f"H+{h}",
            "MAE (Rp)":  f"{mae:,.0f}",
            "RMSE (Rp)": f"{rmse:,.0f}",
            "MAPE (%)":  f"{mape:.2f}%",
            "R²":        f"{r2:.4f}",
            "n sampel":  len(hd),
        })
    if met_rows:
        st.dataframe(pd.DataFrame(met_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REKOMENDASI DISTRIBUSI
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class="section-header">Rekomendasi Jalur Distribusi</div>
    <div class="section-sub">{komoditas} · H+{horizon} · {selected_date} ·
        Jarak maks {THRESHOLD_KM} km
    </div>
    """, unsafe_allow_html=True)

    recs = get_recommendations(df_cls, selected_date, horizon, osrm, THRESHOLD_KM)

    if recs.empty:
        st.warning("Tidak ada rekomendasi untuk kombinasi ini. Coba horizon atau tanggal lain.")
    else:
        inf_col, stat_col = st.columns([3, 1])
        with inf_col:
            st.info(
                f"Ditemukan **{len(recs)} jalur distribusi** berdasarkan prediksi H+{horizon}. "
                f"Rekomendasi dibuat **{horizon} hari sebelumnya**, memberi waktu persiapan logistik."
            )
        with stat_col:
            st.metric("Rerata Selisih Harga", f"Rp {recs['Selisih Harga'].mean():,.0f}")

        # Peta jaringan — mapbox style terang
        st.markdown("""
        <div class="section-header" style="font-size:13px; margin-top:8px;">
            Peta Jalur Rekomendasi
        </div>
        """, unsafe_allow_html=True)

        nodes_in_rec = list(set(recs["Sumber"].tolist() + recs["Tujuan"].tolist()))
        status_lkp   = {r["kab_kota"]: r["status"] for _, r in snapshot.iterrows()}

        edge_lon, edge_lat = [], []
        for _, row in recs.iterrows():
            sc = coord_dict.get(row["Sumber"])
            tc = coord_dict.get(row["Tujuan"])
            if sc and tc:
                edge_lon += [sc[0], tc[0], None]
                edge_lat += [sc[1], tc[1], None]

        n_lon, n_lat, n_text, n_color, n_size = [], [], [], [], []
        for nd in nodes_in_rec:
            c = coord_dict.get(nd)
            if not c:
                continue
            n_lon.append(c[0]); n_lat.append(c[1])
            st_ = status_lkp.get(nd, "Harga Rata-rata")
            n_color.append(STATUS_COLOR.get(st_, "#98a2b3"))
            n_size.append(16 if st_ != "Harga Rata-rata" else 10)
            n_text.append(nd.replace("Kabupaten ", "").replace("Kota ", ""))

        fig_net = go.Figure()
        fig_net.add_trace(go.Scattermapbox(
            lon=edge_lon, lat=edge_lat,
            mode="lines",
            line=dict(width=2, color="#1d6ae5"),
            opacity=0.35,
            hoverinfo="none",
            name="Jalur",
        ))
        fig_net.add_trace(go.Scattermapbox(
            lon=n_lon, lat=n_lat,
            mode="markers+text",
            marker=dict(size=n_size, color=n_color,
                        opacity=0.9,
                        allowoverlap=True),
            text=n_text,
            textfont=dict(size=9, color="#344054"),
            textposition="top right",
            hovertext=[f"{nd}<br>{status_lkp.get(nd,'')}"
                       for nd in nodes_in_rec if coord_dict.get(nd)],
            hoverinfo="text",
            name="Kabupaten",
        ))
        fig_net.update_layout(
            mapbox=dict(
                style="carto-positron",          # ← peta terang
                center=dict(lat=-7.54, lon=112.20),
                zoom=7.0,
            ),
            paper_bgcolor="#ffffff",
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Legenda
        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown('<span class="badge-low">● Sumber — Harga Rendah</span>',    unsafe_allow_html=True)
        lc2.markdown('<span class="badge-avg">● Harga Rata-rata</span>',           unsafe_allow_html=True)
        lc3.markdown('<span class="badge-high">● Tujuan — Harga Tinggi</span>',   unsafe_allow_html=True)

        # Tabel detail
        st.markdown("""
        <div class="section-header" style="font-size:13px; margin-top: 16px;">
            Detail Rekomendasi
        </div>
        """, unsafe_allow_html=True)
        display = recs.copy()
        display["Sumber"]        = display["Sumber"].apply(short_name)
        display["Tujuan"]        = display["Tujuan"].apply(short_name)
        display["Harga Sumber"]  = display["Harga Sumber"].map(lambda x: f"Rp {x:,}")
        display["Harga Tujuan"]  = display["Harga Tujuan"].map(lambda x: f"Rp {x:,}")
        display["Selisih Harga"] = display["Selisih Harga"].map(lambda x: f"Rp {x:,}")
        st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALISIS KETIMPANGAN (CoV)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">Analisis Ketimpangan Harga Antarwilayah</div>
    <div class="section-sub">
        Coefficient of Variation (CoV = std/mean) harga lintas 38 kab/kota.
        Nilai tinggi → harga sangat tidak merata.
    </div>
    """, unsafe_allow_html=True)

    cov_h = df_cov[df_cov["horizon"] == horizon].sort_values("forecast_date").copy()

    if cov_h.empty:
        st.warning("Tidak ada data CoV untuk horizon ini.")
    else:
        cov_h["is_holiday"] = False
        for s, e, _ in HOLIDAY_PERIODS:
            mask = (cov_h["forecast_date"] >= s) & (cov_h["forecast_date"] <= e)
            cov_h.loc[mask, "is_holiday"] = True

        hol_cov    = cov_h[cov_h["is_holiday"]]["cov_actual"].mean()
        normal_cov = cov_h[~cov_h["is_holiday"]]["cov_actual"].mean()
        delta      = hol_cov - normal_cov

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CoV Aktual (rata-rata)",   f"{cov_h['cov_actual'].mean():.2f}%")
        m2.metric("CoV Prediksi (rata-rata)", f"{cov_h['cov_pred'].mean():.2f}%")
        m3.metric("CoV saat Hari Libur",      f"{hol_cov:.2f}%",
                  f"+{delta:.2f} pp vs normal" if delta > 0 else f"{delta:.2f} pp vs normal")
        m4.metric("Selisih Libur vs Normal",  f"{delta:+.2f} pp")

        st.markdown("""<div style="border-top: 1px solid #e4e7ec; margin: 16px 0;"></div>""",
                    unsafe_allow_html=True)

        # Time series CoV
        fig_cov = go.Figure()
        for start, end, label in HOLIDAY_PERIODS:
            fig_cov.add_vrect(
                x0=start, x1=end,
                fillcolor="#fde68a", opacity=0.25,
                layer="below", line_width=0,
            )
            fig_cov.add_annotation(
                x=start, y=1.0, yref="paper",
                text=label, showarrow=False,
                font=dict(color="#b45309", size=9),
                xanchor="left", yanchor="bottom", yshift=5,
            )
        fig_cov.add_trace(go.Scatter(
            x=cov_h["forecast_date"], y=cov_h["cov_actual"],
            mode="lines", name="CoV Aktual",
            line=dict(color=C_GREEN, width=2),
            fill="tozeroy", fillcolor="rgba(18,183,106,0.07)",
        ))
        fig_cov.add_trace(go.Scatter(
            x=cov_h["forecast_date"], y=cov_h["cov_pred"],
            mode="lines", name="CoV Prediksi",
            line=dict(color=C_BLUE, width=1.5, dash="dash"),
        ))
        apply_plotly_theme(
            fig_cov, height=360,
            title=dict(text=f"Ketimpangan Harga H+{horizon} — {komoditas}",
                       font=dict(size=13, color="#101828"), x=0),
            yaxis=dict(gridcolor="#f0f1f3", title="CoV (%)", tickformat=".1f",
                       tickfont=dict(size=11, color="#667085"),
                       title_font=dict(size=11, color="#667085")),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

        # Histogram deviasi
        st.markdown(f"""
        <div class="section-header" style="font-size:13px; margin-top: 4px;">
            Distribusi Deviasi Harga vs Rata-rata Provinsi · H+{horizon}
        </div>
        """, unsafe_allow_html=True)
        pct_all = df_cls[df_cls["horizon"] == horizon]["pct_vs_prov"].dropna()
        fig_hist = px.histogram(
            pct_all, nbins=60,
            color_discrete_sequence=[C_BLUE],
            opacity=0.7,
            labels={"value": "Deviasi vs Rata-rata Provinsi (%)", "count": "Frekuensi"},
            height=260,
        )
        fig_hist.add_vline(
            x=0, line_color=C_AMBER, line_width=1.5,
            annotation_text="Rata-rata Provinsi",
            annotation_font_color=C_AMBER,
            annotation_font_size=9,
        )
        apply_plotly_theme(fig_hist, height=260,
                           margin=dict(l=50, r=20, t=10, b=40))
        st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENTRALITAS GRAF
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="section-header">Analisis Sentralitas Jaringan Distribusi Jawa Timur</div>
    <div class="section-sub">
        Degree = banyak koneksi langsung (hub lokal).
        Betweenness = sering menjadi perantara jalur optimal (hub regional).
    </div>
    """, unsafe_allow_html=True)

    thr = st.slider(
        "Threshold jarak (km) untuk membangun edge",
        50, 200, THRESHOLD_KM, step=10,
    )

    with st.spinner("Menghitung sentralitas…"):
        deg_df, btw_df, G = compute_centrality(thr)

    n_nodes  = G.number_of_nodes()
    n_edges  = G.number_of_edges()
    density_ = nx.density(G)
    avg_deg_ = np.mean([d for _, d in G.degree()])

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Node (kab/kota)", n_nodes)
    sc2.metric("Edge (koneksi)",  n_edges)
    sc3.metric("Density Graf",    f"{density_:.3f}")
    sc4.metric("Rata-rata Degree", f"{avg_deg_:.1f}")

    st.markdown("""<div style="border-top: 1px solid #e4e7ec; margin: 16px 0;"></div>""",
                unsafe_allow_html=True)

    col_d, col_b = st.columns(2)

    # — Degree centrality
    with col_d:
        st.markdown("""
        <div style="font-size: 12px; font-weight: 600; color: #667085;
                    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
            Degree Centrality — Hub Lokal
        </div>
        <div style="font-size: 11.5px; color: #98a2b3; margin-bottom: 10px;">
            Banyak koneksi langsung dalam radius threshold
        </div>
        """, unsafe_allow_html=True)
        top_d = deg_df.head(15).copy()
        top_d["label"] = top_d["kab_kota"].apply(short_name)

        fig_d = go.Figure(go.Bar(
            y=top_d["label"],
            x=top_d["degree"],
            orientation="h",
            marker=dict(
                color=top_d["degree"],
                colorscale=[[0, "#dbeafe"], [0.5, "#60a5fa"], [1.0, C_BLUE]],
                showscale=False,
                line=dict(color="#ffffff", width=0.5),
            ),
            text=top_d["degree"].map(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(size=10, color="#475467"),
        ))
        apply_plotly_theme(fig_d, height=420,
                           margin=dict(l=140, r=70, t=10, b=30),
                           xaxis=dict(gridcolor="#f0f1f3", title="Degree Centrality",
                                      tickfont=dict(size=10, color="#344054"),
                                      title_font=dict(size=11, color="#344054")),
                           yaxis=dict(gridcolor="#f0f1f3", autorange="reversed",
                                      tickfont=dict(size=11, color="#344054"),
                                      title_font=dict(size=11, color="#344054")))
        st.plotly_chart(fig_d, use_container_width=True)

    # — Betweenness centrality
    with col_b:
        st.markdown("""
        <div style="font-size: 12px; font-weight: 600; color: #667085;
                    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
            Betweenness Centrality — Hub Regional
        </div>
        <div style="font-size: 11.5px; color: #98a2b3; margin-bottom: 10px;">
            Sering menjadi perantara jalur distribusi optimal
        </div>
        """, unsafe_allow_html=True)
        top_b = btw_df.head(15).copy()
        top_b["label"] = top_b["kab_kota"].apply(short_name)

        fig_b = go.Figure(go.Bar(
            y=top_b["label"],
            x=top_b["betweenness"],
            orientation="h",
            marker=dict(
                color=top_b["betweenness"],
                colorscale=[[0, "#ede9fe"], [0.5, "#a78bfa"], [1.0, C_PURPLE]],
                showscale=False,
                line=dict(color="#ffffff", width=0.5),
            ),
            text=top_b["betweenness"].map(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(size=10, color="#475467"),
        ))
        apply_plotly_theme(fig_b, height=420,
                           margin=dict(l=140, r=70, t=10, b=30),
                           xaxis=dict(gridcolor="#f0f1f3", title="Betweenness Centrality",
                                      tickfont=dict(size=10, color="#344054"),
                                      title_font=dict(size=11, color="#344054")),
                           yaxis=dict(gridcolor="#f0f1f3", autorange="reversed",
                                      tickfont=dict(size=11, color="#344054"),
                                      title_font=dict(size=11, color="#344054")))
        st.plotly_chart(fig_b, use_container_width=True)

    # Interpretasi otomatis
    if not deg_df.empty and not btw_df.empty:
        hub_lokal    = deg_df.iloc[0]
        hub_regional = btw_df.iloc[0]
        st.markdown(f"""
        <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-left: 4px solid #0284c7;
                    border-radius: 10px; padding: 16px 20px; margin-top: 8px; font-size: 13px;">
            <div style="font-weight: 700; color: #0c4a6e; margin-bottom: 8px;">
                Interpretasi Otomatis (threshold {thr} km)
            </div>
            <p style="color: #0c4a6e; margin: 0 0 6px 0;">
                <strong>Hub lokal terkuat:</strong> {hub_lokal['kab_kota']}
                (degree = {hub_lokal['degree']:.3f}) — titik distribusi dengan koneksi
                langsung terbanyak dalam radius {thr} km.
            </p>
            <p style="color: #0c4a6e; margin: 0;">
                <strong>Hub regional terpenting:</strong> {hub_regional['kab_kota']}
                (betweenness = {hub_regional['betweenness']:.3f}) — melewati lebih dari
                {hub_regional['betweenness']*100:.0f}% dari semua jalur distribusi optimal.
                Penguatan infrastruktur logistik di sini berdampak paling luas.
            </p>
        </div>
        """, unsafe_allow_html=True)
