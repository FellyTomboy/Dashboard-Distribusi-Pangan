import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import math

INPUT_FILE  = "harga_bawang_merah_2018-2025_long.csv"
OUTPUT_FILE = "data final/harga_bawang_merah_features_2018-2025.csv"
COORD_FILE  = "Data Lain-Lain/koordinat_jatim.csv"
R_EARTH_KM  = 6371.0  # radius bumi untuk konversi haversine


# ============================================================
# SPATIAL HELPER: load koordinat & build neighbor tree
# ============================================================

def load_coordinates() -> pd.DataFrame:
    """Load kabupaten coordinates and compute radian lat/lon."""
    df = pd.read_csv(COORD_FILE)
    df["lat_rad"] = np.radians(df["latitude"])
    df["lon_rad"] = np.radians(df["longitude"])
    return df


def build_knn_map(
    coord_df: pd.DataFrame,
    n_neighbors: int = 5,
) -> dict[str, list[str]]:
    """
    Untuk setiap kab_kota, cari n_neighbors tetangga terdekat
    berdasarkan jarak haversine (termasuk dirinya sendiri sebagai
    neighbor pertama -> kita skip index 0 nanti).
    """
    coords = coord_df[["lat_rad", "lon_rad"]].values  # (N, 2)

    # BallTree dengan haversine distance
    tree = BallTree(coords, metric="haversine")

    distances, indices = tree.query(coords, k=n_neighbors + 1)

    names = coord_df["nama"].tolist()
    knn: dict[str, list[str]] = {}

    for i, nama in enumerate(names):
        # indices[i][0] = diri sendiri, pakai 1..n_neighbors
        neighbors = [names[j] for j in indices[i][1:] if j < len(names)]
        knn[nama] = neighbors

    return knn


def compute_spatial_features(
    df: pd.DataFrame,
    coord_df: pd.DataFrame,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Tambahkan fitur spasial:
    - avg_harga_tetangga       : rata-rata harga tetangga
    - selisih_harga_tetangga   : harga_kab - avg_harga_tetangga
    - weighted_prov_price      : rata-rata harga * bobot jarak
    """
    knn_map = build_knn_map(coord_df, n_neighbors=n_neighbors)

    # rename kolom nama agar konsisten
    coord_names = set(coord_df["nama"].tolist())

    # Pastikan hanya kab_kota yang punya koordinat
    df = df[df["kab_kota"].isin(coord_names)].copy()

    neighbor_features = []

    for tanggal, grp in df.groupby("tanggal"):
        #
        # 1. Harga tetangga rata-rata (unweighted)
        #
        price_map: dict[str, float] = {
            row["kab_kota"]: row["lag1_rata_harga"]
            for _, row in grp.iterrows()
        }

        avg_neighbor = {}
        for kab in grp["kab_kota"]:
            neighbors = knn_map.get(kab, [])
            neighbor_prices = [
                price_map.get(n, np.nan)
                for n in neighbors
                if n in price_map
            ]
            valid = [p for p in neighbor_prices if not (pd.isna(p) or p == 0)]
            if valid:
                avg_neighbor[kab] = np.mean(valid)
            else:
                avg_neighbor[kab] = np.nan

        #
        # 2. Weighted province price (inverse-distance weighted)
        #
        # Ambil semua kab_kota di provinsi yang sama (seluruh Jatim)
        all_kabs = list(price_map.keys())
        all_lats = coord_df.set_index("nama").loc[all_kabs, "lat_rad"].to_dict()
        all_lons = coord_df.set_index("nama").loc[all_kabs, "lon_rad"].to_dict()

        weighted_price = {}
        for kab in all_kabs:
            lat1 = all_lats[kab]
            lon1 = all_lons[kab]
            prices = []
            weights = []
            for other in all_kabs:
                if other == kab:
                    continue
                p = price_map.get(other)
                if pd.isna(p) or p == 0:
                    continue
                lat2 = all_lats[other]
                lon2 = all_lons[other]
                # haversine distance
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                )
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                dist_km = R_EARTH_KM * c
                # weight = 1 / (dist + epsilon)
                w = 1.0 / (dist_km + 1.0)
                prices.append(p)
                weights.append(w)
            if prices:
                weighted_price[kab] = np.average(prices, weights=weights)
            else:
                weighted_price[kab] = np.nan

        # Gabungkan ke dataframe group
        grp = grp.copy()
        grp["avg_harga_tetangga"] = grp["kab_kota"].map(avg_neighbor)
        grp["selisih_harga_tetangga"] = (
            grp["lag1_rata_harga"] - grp["avg_harga_tetangga"]
        )
        grp["weighted_prov_price"] = grp["kab_kota"].map(weighted_price)

        neighbor_features.append(grp)

    if neighbor_features:
        result = pd.concat(neighbor_features, ignore_index=True)
        # Gabungkan kembali dengan baris yang tidak punya koordinat (jika ada)
        no_coord = df[~df["kab_kota"].isin(coord_names)]
        for col in ["avg_harga_tetangga", "selisih_harga_tetangga", "weighted_prov_price"]:
            no_coord[col] = np.nan
        result = pd.concat([result, no_coord], ignore_index=True)
    else:
        result = df
        for col in ["avg_harga_tetangga", "selisih_harga_tetangga", "weighted_prov_price"]:
            result[col] = np.nan

    return result.sort_values(["kab_kota", "tanggal"]).reset_index(drop=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)

    df["tanggal"] = pd.to_datetime(df["tanggal"])

    df.sort_values(
        ["kab_kota", "tanggal"],
        inplace=True
    )

    return df


# ============================================================
# FEATURE ENGINEERING PER KAB/KOTA (NO LEAKAGE)
# ============================================================

def feature_engineer_kab(g: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering per kab_kota.

    === ANTI-LEAKAGE RULES ===
    Semua rolling/lag:
      - shift(1) dulu → baru rolling
    Jadi:
      - rolling hanya memakai data <= t-1
      - TIDAK termasuk harga hari t

    Contoh:
      mean_7d untuk target 10 Januari
        = rata-rata 3,4,5,6,7,8,9 Januari   ✓
        = TIDAK memakai 10 Januari            ✗
    """

    g = g.copy().sort_values("tanggal")

    # ============================================================
    # HANDLE MISSING VALUES — forward fill harga sebelum FE
    # Scraping kadang menghasilkan gap 1-8 hari (misal libur/scraper error)
    # Forward fill lebih baik daripada membiarkan NaN merambat ke lag/rolling
    # ============================================================

    missing_count = g["rata_harga"].isna().sum()
    if missing_count > 0:
        print(f"    [INFO] {g['kab_kota'].iloc[0]}: "
              f"forward fill {missing_count} missing values")
        g["rata_harga"] = g["rata_harga"].ffill()

    # ============================================================
    # BASE SERIES (HANYA MASA LALU)
    # ============================================================

    price = g["rata_harga"]
    lag1_price = price.shift(1)  # t-1 → BASE UNTUK SEMUA ROLLING

    # ============================================================
    # 1. LOG TRANSFORM
    # ============================================================

    g["log_lag1_rata_harga"] = np.log(lag1_price + 1)

    # ============================================================
    # 2. LAG FEATURES (SEMUA DARI SHIFT)
    # ============================================================

    g["lag1_rata_harga"] = lag1_price
    g["lag2_rata_harga"] = price.shift(2)
    g["lag3_rata_harga"] = price.shift(3)

    # --- NEW: Weekly lags ---
    g["lag7_rata_harga"]  = price.shift(7)
    g["lag14_rata_harga"] = price.shift(14)
    g["lag21_rata_harga"] = price.shift(21)
    g["lag30_rata_harga"] = price.shift(30)

    # ============================================================
    # 3. PRICE CHANGE
    # ============================================================

    g["delta_harga"] = (
        lag1_price - price.shift(2)
    )

    g["pct_change_harga"] = (
        g["delta_harga"]
        / (price.shift(2) + 1e-6)
    )

    # ============================================================
    # 4. ROLLING FEATURES — PAST ONLY (via lag1_price = t-1)
    #    Rolling di sini hanya menjangkau t-1, t-2, t-3, ...
    #    TIDAK PERNAH menyentuh harga hari t.
    # ============================================================

    # --- Existing 3d rolling ---
    g["harga_3d_mean"] = lag1_price.rolling(3, min_periods=1).mean()
    g["harga_3d_min"]  = lag1_price.rolling(3, min_periods=1).min()
    g["harga_3d_max"]  = lag1_price.rolling(3, min_periods=1).max()
    g["volatilitas_3d"] = lag1_price.rolling(3, min_periods=1).std()

    # --- NEW: longer rolling windows ---
    g["harga_7d_mean"]  = lag1_price.rolling(7, min_periods=1).mean()
    g["harga_7d_std"]   = lag1_price.rolling(7, min_periods=1).std()
    g["harga_7d_min"]   = lag1_price.rolling(7, min_periods=1).min()
    g["harga_7d_max"]   = lag1_price.rolling(7, min_periods=1).max()

    g["harga_14d_mean"] = lag1_price.rolling(14, min_periods=1).mean()
    g["harga_30d_mean"] = lag1_price.rolling(30, min_periods=1).mean()

    # ============================================================
    # 5. MOMENTUM / TREND STRENGTH
    # ============================================================

    # --- Existing ---
    g["harga_trend_3d"] = (
        g["lag1_rata_harga"] - g["harga_3d_mean"]
    )

    g["is_price_up"] = (
        g["delta_harga"] > 0
    ).astype(int)

    # --- NEW: momentum ---
    g["momentum_vs_7d"] = (
        g["lag1_rata_harga"] - g["harga_7d_mean"]
    )

    g["momentum_7d_vs_30d"] = (
        g["harga_7d_mean"] - g["harga_30d_mean"]
    )

    # --- NEW: Short-term vs long-term trend ---
    g["trend_7d"] = g["harga_7d_mean"] - g["harga_14d_mean"]
    g["trend_14d"] = g["harga_14d_mean"] - g["harga_30d_mean"]

    # ============================================================
    # 6. Z-SCORE LOKAL
    # ============================================================

    rolling_mean_7d = lag1_price.rolling(7, min_periods=3).mean()
    rolling_std_7d  = lag1_price.rolling(7, min_periods=3).std()

    g["zscore_harga_7d"] = (
        (g["lag1_rata_harga"] - rolling_mean_7d)
        / (rolling_std_7d + 1e-6)
    )

    return g


# ============================================================
# FEATURE PROVINSI — LEAVE-ONE-OUT (ANTI-LEAKAGE)
# ============================================================

def feature_engineer_provinsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur lintas kabupaten dengan LEAVE-ONE-OUT provinsi mean.

    === KENAPA INI PENTING ===
    Tanpa leave-one-out:
        rata_harga_provinsi untuk Kab. Bangkalan
        = rata-rata SEMUA kabupaten (termasuk Bangkalan sendiri)

    Masalah:
        model akan "melihat" harga Bangkalan
        melalui rata_harga_provinsi → LEAKAGE RINGAN

    Solusi LEAVE-ONE-OUT:
        Untuk Kab. Bangkalan, rata_harga_provinsi
        = rata-rata SEMUA kabupaten LAIN (tanpa Bangkalan)

    === IMPLEMENTASI ===
    Semua tetap berbasis lag1_price (t-1) → aman.
    """

    # ============================================================
    # LEAVE-ONE-OUT PROVINCE MEAN
    # ============================================================

    # Total sum & count per tanggal
    prov_agg = (
        df.groupby("tanggal")["lag1_rata_harga"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "prov_sum", "count": "prov_count"})
    )

    df = df.merge(prov_agg, on="tanggal", how="left")

    # Leave-one-out: (total_sum - self) / (total_count - 1)
    self_val = df["lag1_rata_harga"].fillna(0)
    df["rata_harga_provinsi_loo"] = (
        (df["prov_sum"] - self_val)
        / (df["prov_count"] - 1)
    )
    # Jika hanya 1 kabupaten → fallback ke mean biasa
    df.loc[df["prov_count"] <= 1, "rata_harga_provinsi_loo"] = (
        df.loc[df["prov_count"] <= 1, "lag1_rata_harga"]
    )

    # ============================================================
    # LEAVE-ONE-OUT MIN / MAX / STD
    # ============================================================

    # --- MAX tanpa self ---
    def loo_max(grp):
        vals = grp.values
        n = len(vals)
        out = np.full(n, np.nan)
        for i in range(n):
            mask = np.arange(n) != i
            out[i] = np.nanmax(vals[mask]) if n > 1 else vals[i]
        return out

    df["max_harga_provinsi_loo"] = (
        df.groupby("tanggal")["lag1_rata_harga"]
        .transform(loo_max)
    )

    # --- MIN tanpa self ---
    def loo_min(grp):
        vals = grp.values
        n = len(vals)
        out = np.full(n, np.nan)
        for i in range(n):
            mask = np.arange(n) != i
            out[i] = np.nanmin(vals[mask]) if n > 1 else vals[i]
        return out

    df["min_harga_provinsi_loo"] = (
        df.groupby("tanggal")["lag1_rata_harga"]
        .transform(loo_min)
    )

    # --- STD tanpa self ---
    def loo_std(grp):
        vals = grp.values
        n = len(vals)
        out = np.full(n, np.nan)
        for i in range(n):
            mask = np.arange(n) != i
            out[i] = np.nanstd(vals[mask]) if n > 1 else 0.0
        return out

    df["std_harga_provinsi_loo"] = (
        df.groupby("tanggal")["lag1_rata_harga"]
        .transform(loo_std)
    )

    # ============================================================
    # DEVIASI DARI PROVINSI (berdasarkan LOO mean)
    # ============================================================

    df["deviasi_rata_harga"] = (
        df["lag1_rata_harga"]
        - df["rata_harga_provinsi_loo"]
    )

    df["pct_diff_harga_provinsi"] = (
        df["deviasi_rata_harga"]
        / (df["rata_harga_provinsi_loo"] + 1e-6)
    )

    # ============================================================
    # RANK HARGA
    # ============================================================

    df["rank_harga_kab"] = (
        df.groupby("tanggal")["lag1_rata_harga"]
        .rank(method="min", ascending=True)
    )

    # ============================================================
    # HAPUS KOLOM BANTU
    # ============================================================

    df.drop(columns=["prov_sum", "prov_count"], inplace=True)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print(f"Memuat data: {INPUT_FILE}")

    df = load_data()

    print(
        f"  {len(df)} baris | "
        f"{df['kab_kota'].nunique()} kab/kota | "
        f"{df['tanggal'].nunique()} hari"
    )

    # ============================================================
    # FEATURE PER KAB/KOTA
    # ============================================================

    print("Membuat feature per kab_kota...")

    groups = []

    for nama, grp in df.groupby("kab_kota"):
        groups.append(feature_engineer_kab(grp))

    df = pd.concat(groups, ignore_index=True)

    # ============================================================
    # FEATURE PROVINSI (leave-one-out)
    # ============================================================

    print("Membuat feature provinsi (leave-one-out)...")

    df = feature_engineer_provinsi(df)

    # ============================================================
    # FEATURE SPASIAL (berdasarkan koordinat)
    # ============================================================

    print("Membuat feature spasial (tetangga geografis)...")

    coord_df = load_coordinates()
    df = compute_spatial_features(df, coord_df, n_neighbors=5)

    # ============================================================
    # OUTPUT
    # ============================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSelesai!")
    print(f"Output: {OUTPUT_FILE}")

    print(f"\nJumlah kolom: {len(df.columns)}")

    # Tampilkan kolom baru
    new_cols = [
        "lag7_rata_harga", "lag14_rata_harga", "lag21_rata_harga", "lag30_rata_harga",
        "harga_7d_mean", "harga_7d_std", "harga_7d_min", "harga_7d_max",
        "harga_14d_mean", "harga_30d_mean",
        "momentum_vs_7d", "momentum_7d_vs_30d", "trend_7d", "trend_14d",
        "rata_harga_provinsi_loo", "min_harga_provinsi_loo",
        "max_harga_provinsi_loo", "std_harga_provinsi_loo",
        "avg_harga_tetangga", "selisih_harga_tetangga", "weighted_prov_price",
    ]
    existing_new = [c for c in new_cols if c in df.columns]
    print(f"\nKolom baru ({len(existing_new)}):")
    for c in existing_new:
        print(f"  + {c}")

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
