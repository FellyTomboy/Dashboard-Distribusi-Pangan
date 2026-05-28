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
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── TEMA & CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp                    { background: #080d18; color: #dde1ef; }
section[data-testid="stSidebar"] { background: #0c1222; border-right: 1px solid #162035; }
[data-testid="stMetricValue"]    { font-size: 1.4rem !important; font-weight: 700; color: #e2e8f0; }
[data-testid="stMetricLabel"]    { font-size: 0.75rem; color: #6b7280; }
[data-testid="stMetricDelta"]    { font-size: 0.78rem !important; }

.stTabs [data-baseweb="tab-list"] { background: #0c1222; gap: 2px; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"]      { color: #6b7280 !important; border-radius: 8px; padding: 8px 16px; font-size: 0.85rem; }
.stTabs [aria-selected="true"]    { background: #162035 !important; color: #93c5fd !important; font-weight: 600; }

.card {
    background: linear-gradient(145deg, #0f1a2e 0%, #162035 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.badge-high { color: #f87171; font-weight: 700; }
.badge-low  { color: #4ade80; font-weight: 700; }
.badge-avg  { color: #fbbf24; font-weight: 700; }

h1, h2, h3, h4 { color: #dde1ef !important; }
p, li, span    { color: #9ca3af; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

div[data-testid="stDataFrame"] table { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── KONSTANTA ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
THRESHOLD_KM = 120

STATUS_COLOR = {
    "Harga Rendah":    "#4ade80",
    "Harga Rata-rata": "#fbbf24",
    "Harga Tinggi":    "#f87171",
    "Tidak Ada Data":  "#374151",
}

HOLIDAY_PERIODS = [
    ("2024-03-28", "2024-04-16", "Lebaran 2024"),
    ("2024-12-22", "2025-01-05", "Nataru 2024/25"),
    ("2025-03-17", "2025-04-08", "Lebaran 2025"),
    ("2025-12-22", "2025-12-31", "Nataru 2025/26"),
]

# ── UTILITAS ────────────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    """Normalisasi nama kab/kota → huruf besar tanpa prefiks KABUPATEN/KOTA."""
    n = str(name).upper().strip()
    for prefix in ("KABUPATEN ", "KOTA "):
        n = n.replace(prefix, "")
    return n


def short_name(name: str) -> str:
    return name.replace("Kabupaten ", "Kab. ").replace("Kota ", "Kota ")


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
    """Muat GeoJSON dan filter hanya Jawa Timur (province_id='35')."""
    with open(BASE / "all_kabkota_ind.geojson", encoding="utf-8") as f:
        gj = json.load(f)
    jatim = [
        feat for feat in gj["features"]
        if feat["properties"].get("province_id") == "35"
    ]
    # Tambahkan norm_name sebagai id fitur agar cocok dengan dataframe
    for feat in jatim:
        norm = normalize_name(feat["properties"]["name"])
        feat["id"] = norm
        feat["properties"]["norm_name"] = norm
    return {"type": "FeatureCollection", "features": jatim}


# ── FUNGSI KALKULASI ─────────────────────────────────────────────────────────────
@st.cache_data
def compute_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung status harga (Rendah/Rata-rata/Tinggi) berdasarkan threshold persentil
    per tanggal+horizon.  Hasilnya adalah dataframe dengan kolom tambahan:
    status, prov_mean, pct_vs_prov.
    """
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
    """CoV (std/mean) harga per tanggal+horizon melintasi 38 kab/kota."""
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
    """
    Bangun graf dari data OSRM dan hitung degree + betweenness centrality.
    Return: (deg_df, btw_df) — sudah diurutkan descending.
    """
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
    """Rekomendasikan jalur distribusi langsung sumber→tujuan untuk hari+horizon tertentu."""
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
                "Sumber":           src["kab_kota"],
                "Tujuan":           tgt["kab_kota"],
                "Harga Sumber":     int(src["prediction"]),
                "Harga Tujuan":     int(tgt["prediction"]),
                "Selisih Harga":    int(selisih),
                "Jarak (km)":       round(e["jarak_km"], 1),
                "Waktu (menit)":    round(e["waktu_menit"], 0),
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
coords = load_coords()
osrm   = load_osrm()
geojson = load_geojson()

# Buat dict koordinat
coord_dict: dict[str, tuple[float, float]] = {
    row["nama"]: (row["longitude"], row["latitude"])
    for _, row in coords.iterrows()
}

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 4px 0;">
        <span style="font-size: 1.4rem;">🌾</span>
        <span style="font-size: 1.1rem; font-weight: 800; color: #93c5fd;"> Dashboard Distribusi Pangan </span>
        <p style="font-size: 0.72rem; color: #4b5563; margin: 2px 0 16px 0;">
             Jawa Timur
        </p>
    </div>
    """, unsafe_allow_html=True)

    komoditas = st.selectbox(
        "🌶️ Komoditas",
        ["Cabai Rawit Merah", "Bawang Merah"],
    )
    st.markdown("---")

    # Load & hitung
    df_raw = load_predictions(komoditas)
    df_cls = compute_classification(df_raw)
    df_cov = compute_cov(df_cls)

    available_dates = sorted(df_raw["forecast_date"].dt.date.unique())

    # Date input — lebih ringan dari select_slider untuk 700+ tanggal
    min_d, max_d = available_dates[0], available_dates[-1]
    selected_date = st.date_input(
        "📅 Tanggal Prediksi",
        value=min_d,
        min_value=min_d,
        max_value=max_d,
    )

    horizon = st.select_slider(
        "🔭 Horizon Prediksi",
        options=[1, 2, 3, 4, 5, 6, 7],
        value=1,
        format_func=lambda x: f"H+{x}",
    )

    st.markdown("---")

    # Statistik cepat untuk sidebar
    snapshot = df_cls[
        (df_cls["forecast_date"] == pd.Timestamp(selected_date)) &
        (df_cls["horizon"] == horizon)
    ]

    if not snapshot.empty:
        n_high = (snapshot["status"] == "Harga Tinggi").sum()
        n_low  = (snapshot["status"] == "Harga Rendah").sum()
        n_avg  = (snapshot["status"] == "Harga Rata-rata").sum()
        prov_mean = snapshot["prediction"].mean()

        st.markdown(f"**Status Wilayah · H+{horizon}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴", n_high, "Tinggi")
        c2.metric("🟢", n_low,  "Rendah")
        c3.metric("🟡", n_avg,  "Rata²")
        st.metric("Rata-rata Provinsi", f"Rp {prov_mean:,.0f}")

    st.markdown("---")
    st.caption(
        "Model: LightGBM + Recursive Multi-Step Forecasting  \n"
        "Data: SISKAPERBAPO · Open-Meteo · OSRM  \n"
        "Periode uji: Jan 2024 – Des 2025  \n\n"
        "*Arsitektur realtime: pengembangan lanjutan.*"
    )

# ── HEADER ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 4px 0 20px 0;">
    <h1 style="font-size: 1.65rem; font-weight: 800; margin: 0;
               background: linear-gradient(90deg, #93c5fd 0%, #4ade80 60%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🌾 Sistem Prediksi Harga & Rekomendasi Distribusi Pangan Jawa Timur
    </h1>
    <p style="color: #6b7280; margin: 6px 0 0 0; font-size: 0.82rem;">
        {komoditas} · 38 Kabupaten/Kota · Tampilan: H+{horizon} · {selected_date}
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Peta Harga",
    "📈  Prediksi per Wilayah",
    "🔀  Rekomendasi Distribusi",
    "📊  Analisis Ketimpangan",
    "🏆  Sentralitas Graf",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 ─ PETA HARGA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"#### 🗺️ Peta Status Harga — {komoditas} · H+{horizon} · {selected_date}")

    if snapshot.empty:
        st.warning("Tidak ada data untuk tanggal ini. Coba pilih tanggal lain.")
    else:
        # Buat dataframe untuk choropleth dengan norm_name
        map_df = snapshot.copy()
        map_df["norm_name"] = map_df["kab_kota"].apply(normalize_name)
        map_df["color_num"] = map_df["status"].map(
            {"Harga Rendah": 0, "Harga Rata-rata": 1, "Harga Tinggi": 2}
        )
        map_df["Prediksi (Rp)"] = map_df["prediction"].map(lambda x: f"Rp {x:,.0f}")
        map_df["Aktual (Rp)"]   = map_df["actual"].map(lambda x: f"Rp {x:,.0f}")
        map_df["Deviasi (%)"]   = map_df["pct_vs_prov"].map(lambda x: f"{x:+.1f}%")

        fig_map = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="norm_name",
            featureidkey="id",
            color="color_num",
            color_continuous_scale=[
                [0.0,  "#1a4731"],
                [0.25, "#4ade80"],
                [0.50, "#fbbf24"],
                [0.75, "#f87171"],
                [1.0,  "#7f1d1d"],
            ],
            range_color=[0, 2],
            mapbox_style="carto-darkmatter",
            center={"lat": -7.54, "lon": 112.20},
            zoom=7.0,
            opacity=0.78,
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
            paper_bgcolor="#080d18",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Legenda
        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown('<span class="badge-low">●</span> Harga Rendah — sumber distribusi', unsafe_allow_html=True)
        lc2.markdown('<span class="badge-avg">●</span> Harga Rata-rata — kondisi normal', unsafe_allow_html=True)
        lc3.markdown('<span class="badge-high">●</span> Harga Tinggi — prioritas tujuan', unsafe_allow_html=True)

        st.markdown("---")

        # Tabel ringkasan
        st.markdown("##### Ringkasan Status 38 Kabupaten/Kota")
        disp = (
            map_df[["kab_kota", "Prediksi (Rp)", "Aktual (Rp)", "status", "Deviasi (%)"]]
            .rename(columns={"kab_kota": "Kabupaten/Kota", "status": "Status"})
            .sort_values("Status")
        )
        st.dataframe(disp, use_container_width=True, hide_index=True, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 ─ PREDIKSI PER WILAYAH
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📈 Grafik Prediksi H+1–H+7 per Wilayah")

    all_kabs = sorted(df_raw["kab_kota"].unique())
    sel_kab  = st.selectbox("Pilih Kabupaten/Kota", all_kabs, key="kab_tab2")

    kab_data = df_raw[df_raw["kab_kota"] == sel_kab]
    fan_data = kab_data[kab_data["forecast_date"] == pd.Timestamp(selected_date)]

    if fan_data.empty:
        st.info("Tidak ada data prediksi untuk kabupaten dan tanggal yang dipilih.")
    else:
        col_fan, col_ts = st.columns([1, 2])

        # — Fan chart prediksi H+1..H+7 dari tanggal terpilih
        with col_fan:
            st.markdown(f"**Prediksi dari {selected_date}**")
            fig_fan = go.Figure()
            fig_fan.add_trace(go.Scatter(
                x=[f"H+{h}" for h in fan_data["horizon"]],
                y=fan_data["prediction"],
                mode="lines+markers",
                name="Prediksi",
                line=dict(color="#93c5fd", width=2.5),
                marker=dict(size=9),
            ))
            fig_fan.add_trace(go.Scatter(
                x=[f"H+{h}" for h in fan_data["horizon"]],
                y=fan_data["actual"],
                mode="lines+markers",
                name="Aktual",
                line=dict(color="#4ade80", width=2, dash="dash"),
                marker=dict(size=9, symbol="diamond"),
            ))
            fig_fan.update_layout(
                paper_bgcolor="#0f1a2e", plot_bgcolor="#0f1a2e",
                font=dict(color="#dde1ef", size=11),
                height=300,
                margin=dict(l=50, r=10, t=10, b=40),
                legend=dict(orientation="h", y=1.05),
                xaxis=dict(gridcolor="#162035"),
                yaxis=dict(gridcolor="#162035", tickformat=",.0f", title="Harga (Rp)"),
            )
            st.plotly_chart(fig_fan, use_container_width=True)

        # — Time series H+1 sepanjang periode
        with col_ts:
            st.markdown("**Tren Harga H+1 Sepanjang Periode Uji**")
            h1 = kab_data[kab_data["horizon"] == 1].sort_values("forecast_date")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=h1["forecast_date"], y=h1["actual"],
                mode="lines", name="Aktual",
                line=dict(color="#4ade80", width=1.5),
                fill="tozeroy", fillcolor="rgba(74,222,128,0.06)",
            ))
            fig_ts.add_trace(go.Scatter(
                x=h1["forecast_date"], y=h1["prediction"],
                mode="lines", name="Prediksi H+1",
                line=dict(color="#93c5fd", width=1.5),
            ))
            for start, _, label in HOLIDAY_PERIODS:
                # 1. Buat garis vertikalnya saja
                fig_ts.add_vline(
                    x=start,
                    line_width=1,
                    line_dash="dot",
                    line_color="#fbbf24"
                )
                # 2. Buat anotasinya secara terpisah
                fig_ts.add_annotation(
                    x=start,
                    y=1.0,
                    yref="paper",
                    text=label,
                    showarrow=False,
                    font=dict(color="#fbbf24", size=8),
                    xanchor="left",
                    yanchor="bottom"
                )
            fig_ts.update_layout(
                paper_bgcolor="#0f1a2e", plot_bgcolor="#0f1a2e",
                font=dict(color="#dde1ef", size=11),
                height=300,
                margin=dict(l=50, r=10, t=10, b=40),
                legend=dict(orientation="h", y=1.05),
                xaxis=dict(gridcolor="#162035"),
                yaxis=dict(gridcolor="#162035", tickformat=",.0f", title="Harga (Rp)"),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    # Tabel metrik akurasi per horizon untuk kabupaten terpilih
    st.markdown("---")
    st.markdown("##### Metrik Akurasi Model per Horizon")
    met_rows = []
    for h in range(1, 8):
        hd = kab_data[kab_data["horizon"] == h].dropna(subset=["actual", "prediction"])
        if len(hd) < 2:
            continue
        err = hd["prediction"] - hd["actual"]
        mae  = np.abs(err).mean()
        rmse = np.sqrt((err ** 2).mean())
        mape = (np.abs(err / hd["actual"]) * 100).mean()
        ss_res = (err ** 2).sum()
        ss_tot = ((hd["actual"] - hd["actual"].mean()) ** 2).sum()
        r2 = max(1 - ss_res / ss_tot if ss_tot > 0 else 0, 0)
        met_rows.append({
            "Horizon": f"H+{h}",
            "MAE (Rp)": f"{mae:,.0f}",
            "RMSE (Rp)": f"{rmse:,.0f}",
            "MAPE (%)": f"{mape:.2f}%",
            "R²": f"{r2:.4f}",
            "n sampel": len(hd),
        })
    if met_rows:
        st.dataframe(pd.DataFrame(met_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 ─ REKOMENDASI DISTRIBUSI
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f"#### 🔀 Rekomendasi Jalur Distribusi — {komoditas} · H+{horizon} · {selected_date}"
    )

    recs = get_recommendations(df_cls, selected_date, horizon, osrm, THRESHOLD_KM)

    if recs.empty:
        st.warning(
            "Tidak ada rekomendasi untuk kombinasi ini. Coba horizon atau tanggal lain."
        )
    else:
        inf_col, stat_col = st.columns([3, 1])
        with inf_col:
            st.info(
                f"💡 Ditemukan **{len(recs)} jalur distribusi** berdasarkan prediksi H+{horizon}. "
                f"Rekomendasi dibuat **{horizon} hari sebelumnya**, memberi waktu persiapan logistik."
            )
        with stat_col:
            st.metric("Rerata Selisih Harga", f"Rp {recs['Selisih Harga'].mean():,.0f}")

        # Visualisasi jaringan distribusi
        st.markdown("##### Peta Jalur Rekomendasi")

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
            n_lon.append(c[0])
            n_lat.append(c[1])
            st_ = status_lkp.get(nd, "Harga Rata-rata")
            n_color.append(STATUS_COLOR.get(st_, "#374151"))
            n_size.append(16 if st_ != "Harga Rata-rata" else 10)
            n_text.append(nd.replace("Kabupaten ", "").replace("Kota ", ""))

        fig_net = go.Figure()
        fig_net.add_trace(go.Scattermapbox(
            lon=edge_lon, lat=edge_lat,
            mode="lines",
            line=dict(width=1.5, color="#93c5fd"),
            opacity=0.55,
            hoverinfo="none",
            name="Jalur",
        ))
        fig_net.add_trace(go.Scattermapbox(
            lon=n_lon, lat=n_lat,
            mode="markers+text",
            marker=dict(size=n_size, color=n_color),
            text=n_text,
            textfont=dict(size=8, color="#dde1ef"),
            textposition="top right",
            hovertext=[f"{nd}<br>{status_lkp.get(nd,'')}" for nd in nodes_in_rec if coord_dict.get(nd)],
            hoverinfo="text",
            name="Kabupaten",
        ))
        fig_net.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=-7.54, lon=112.20),
                zoom=7.0,
            ),
            paper_bgcolor="#080d18",
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Legenda warna node
        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown('<span class="badge-low">●</span> Sumber (Harga Rendah)', unsafe_allow_html=True)
        lc2.markdown('<span class="badge-avg">●</span> Harga Rata-rata', unsafe_allow_html=True)
        lc3.markdown('<span class="badge-high">●</span> Tujuan (Harga Tinggi)', unsafe_allow_html=True)

        # Tabel detail
        st.markdown("##### Detail Rekomendasi")
        display = recs.copy()
        display["Sumber"]        = display["Sumber"].apply(short_name)
        display["Tujuan"]        = display["Tujuan"].apply(short_name)
        display["Harga Sumber"]  = display["Harga Sumber"].map(lambda x: f"Rp {x:,}")
        display["Harga Tujuan"]  = display["Harga Tujuan"].map(lambda x: f"Rp {x:,}")
        display["Selisih Harga"] = display["Selisih Harga"].map(lambda x: f"Rp {x:,}")
        st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 ─ ANALISIS KETIMPANGAN (CoV)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📊 Analisis Ketimpangan Harga Antarwilayah (Coefficient of Variation)")
    st.caption(
        "CoV = std/mean harga lintas 38 kab/kota. Nilai tinggi → harga sangat tidak merata antarwilayah."
    )

    cov_h = df_cov[df_cov["horizon"] == horizon].sort_values("forecast_date").copy()

    if cov_h.empty:
        st.warning("Tidak ada data CoV untuk horizon ini.")
    else:
        # Tandai periode libur
        cov_h["is_holiday"] = False
        for s, e, _ in HOLIDAY_PERIODS:
            mask = (cov_h["forecast_date"] >= s) & (cov_h["forecast_date"] <= e)
            cov_h.loc[mask, "is_holiday"] = True

        hol_cov    = cov_h[cov_h["is_holiday"]]["cov_actual"].mean()
        normal_cov = cov_h[~cov_h["is_holiday"]]["cov_actual"].mean()
        delta      = hol_cov - normal_cov

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CoV Aktual (rata-rata)",    f"{cov_h['cov_actual'].mean():.2f}%")
        m2.metric("CoV saat Hari Libur",       f"{hol_cov:.2f}%",    f"+{delta:.2f} pp vs normal")
        m3.metric("CoV Prediksi (rata-rata)",  f"{cov_h['cov_pred'].mean():.2f}%")
        m4.metric("Selisih Libur vs Normal",   f"{delta:+.2f} pp")

        st.markdown("---")

        # Time series CoV
        fig_cov = go.Figure()

        for start, end, label in HOLIDAY_PERIODS:
            # 1. Buat blok warnanya saja
            fig_cov.add_vrect(
                x0=start,
                x1=end,
                fillcolor="#fbbf24", opacity=0.07,
                layer="below", line_width=0
            )
            # 2. Buat anotasinya secara terpisah
            fig_cov.add_annotation(
                x=start,
                y=1.0,
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(color="#fbbf24", size=9),
                xanchor="left",
                yanchor="bottom",
                yshift=5
            )

        fig_cov.add_trace(go.Scatter(
            x=cov_h["forecast_date"], y=cov_h["cov_actual"],
            mode="lines", name="CoV Aktual",
            line=dict(color="#4ade80", width=1.8),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.07)",
        ))
        fig_cov.add_trace(go.Scatter(
            x=cov_h["forecast_date"], y=cov_h["cov_pred"],
            mode="lines", name="CoV Prediksi",
            line=dict(color="#93c5fd", width=1.5, dash="dash"),
        ))
        fig_cov.update_layout(
            title=dict(
                text=f"Ketimpangan Harga H+{horizon} — {komoditas}",
                font=dict(size=12), x=0,
            ),
            paper_bgcolor="#0f1a2e", plot_bgcolor="#0f1a2e",
            font=dict(color="#dde1ef", size=11),
            height=360,
            margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(orientation="h", y=1.06),
            xaxis=dict(gridcolor="#162035"),
            yaxis=dict(gridcolor="#162035", title="CoV (%)", tickformat=".1f"),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

        # Distribusi deviasi harga vs provinsi
        st.markdown("##### Distribusi Deviasi Harga vs Rata-rata Provinsi (Semua Hari · H+{})".format(horizon))

        pct_all = df_cls[df_cls["horizon"] == horizon]["pct_vs_prov"].dropna()
        fig_hist = px.histogram(
            pct_all, nbins=60,
            color_discrete_sequence=["#60a5fa"],
            opacity=0.75,
            labels={"value": "Deviasi vs Rata-rata Provinsi (%)", "count": "Frekuensi"},
            height=260,
        )
        fig_hist.add_vline(
            x=0, line_color="#fbbf24", line_width=1.5,
            annotation_text="Rata-rata Provinsi",
            annotation_font_color="#fbbf24", annotation_font_size=9,
        )
        fig_hist.update_layout(
            paper_bgcolor="#0f1a2e", plot_bgcolor="#0f1a2e",
            font=dict(color="#dde1ef", size=11),
            margin=dict(l=50, r=20, t=10, b=40),
            xaxis=dict(gridcolor="#162035"),
            yaxis=dict(gridcolor="#162035"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 ─ SENTRALITAS GRAF
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### 🏆 Analisis Sentralitas Jaringan Distribusi Jawa Timur")

    thr = st.slider("Threshold jarak (km) untuk membangun edge", 50, 200, THRESHOLD_KM, step=10)

    with st.spinner("Menghitung sentralitas…"):
        deg_df, btw_df, G = compute_centrality(thr)

    # Statistik graf
    n_nodes  = G.number_of_nodes()
    n_edges  = G.number_of_edges()
    density_ = nx.density(G)
    avg_deg_ = np.mean([d for _, d in G.degree()])

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Node (kab/kota)", n_nodes)
    sc2.metric("Edge (koneksi)", n_edges)
    sc3.metric("Density Graf", f"{density_:.3f}")
    sc4.metric("Rata-rata Degree", f"{avg_deg_:.1f}")

    st.markdown("---")

    col_d, col_b = st.columns(2)

    # — Degree centrality
    with col_d:
        st.markdown("##### Degree Centrality")
        st.caption("Banyak koneksi langsung = hub distribusi **lokal**")
        top_d = deg_df.head(15).copy()
        top_d["label"] = top_d["kab_kota"].apply(short_name)

        fig_d = go.Figure(go.Bar(
            y=top_d["label"],
            x=top_d["degree"],
            orientation="h",
            marker=dict(
                color=top_d["degree"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#93c5fd"], [1, "#4ade80"]],
                showscale=False,
            ),
            text=top_d["degree"].map(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(size=9, color="#dde1ef"),
        ))
        fig_d.update_layout(
            paper_bgcolor="#0f1a2e", plot_bgcolor="#0f1a2e",
            font=dict(color="#dde1ef", size=10),
            height=420,
            margin=dict(l=130, r=60, t=10, b=30),
            xaxis=dict(gridcolor="#162035", title="Degree Centrality"),
            yaxis=dict(gridcolor="#162035", autorange="reversed"),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # — Betweenness centrality
    with col_b:
        st.markdown("##### Betweenness Centrality")
        st.caption("Sering menjadi perantara jalur optimal = hub distribusi **regional**")
        top_b = btw_df.head(15).copy()
        top_b["label"] = top_b["kab_kota"].apply(short_name)

        fig_b = go.Figure(go.Bar(
            y=top_b["label"],
            x=top_b["betweenness"],
            orientation="h",
            marker=dict(
                color=top_b["betweenness"],
                colorscale=[[0, "#2d1b4e"], [0.5, "#a855f7"], [1, "#fbbf24"]],
                showscale=False,
            ),
            text=top_b["betweenness"].map(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(size=9, color="#dde1ef"),
        ))
        fig_b.update_layout(
            paper_bgcolor="#0f1a2e", plot_bgcolor="#0f1a2e",
            font=dict(color="#dde1ef", size=10),
            height=420,
            margin=dict(l=130, r=60, t=10, b=30),
            xaxis=dict(gridcolor="#162035", title="Betweenness Centrality"),
            yaxis=dict(gridcolor="#162035", autorange="reversed"),
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # Interpretasi otomatis
    if not deg_df.empty and not btw_df.empty:
        hub_lokal    = deg_df.iloc[0]
        hub_regional = btw_df.iloc[0]
        st.info(
            f"**Interpretasi Otomatis (threshold {thr} km):**\n\n"
            f"- **Hub lokal terkuat**: {hub_lokal['kab_kota']} "
            f"(degree = {hub_lokal['degree']:.3f}) — titik distribusi dengan koneksi langsung terbanyak "
            f"di dalam radius {thr} km.\n"
            f"- **Hub regional terpenting**: {hub_regional['kab_kota']} "
            f"(betweenness = {hub_regional['betweenness']:.3f}) — melewati lebih dari "
            f"{hub_regional['betweenness']*100:.0f}% dari semua jalur distribusi optimal "
            f"di jaringan. Penguatan infrastruktur logistik di sini berdampak paling luas."
        )