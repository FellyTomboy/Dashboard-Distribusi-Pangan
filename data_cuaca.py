import pandas as pd
import numpy as np
import re

INPUT_FILE  = "Data Lain-Lain/cuaca_jatim_2018_2025.csv"
OUTPUT_FILE = "data final/weather_features_2018-2025.csv"

WEATHER_COLS = [
    "temperature_2m_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "temperature_2m_min",
    "vapour_pressure_deficit_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "soil_moisture_0_to_7cm_mean",
]

SHORT_NAMES = {
    "precipitation_sum": "precipitation",
    "et0_fao_evapotranspiration": "et0",
    "vapour_pressure_deficit_max": "vpd",
    "relative_humidity_2m_mean": "rh",
    "wind_speed_10m_max": "wind",
    "shortwave_radiation_sum": "radiation",
    "soil_moisture_0_to_7cm_mean": "soil_moisture",
    "temperature_2m_mean": "temperature",
    "temperature_2m_min": "temperature_min",
}


def clean_col(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s*\(.*?\)", "", name).strip()
    return name

def load_data() -> pd.DataFrame:

    df = pd.read_csv(INPUT_FILE)
    df.columns = [clean_col(c) for c in df.columns]
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df.sort_values(
        ["kab_kota", "tanggal"],
        inplace=True
    )

    return df

def feature_engineer_group(g: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering cuaca per kab_kota.

    === ANTI-LEAKAGE ===
    Semua feature berdasarkan lag1 (data t-1),
    jadi rolling hanya menjangkau t-1, t-2, ...
    TIDAK PERNAH menyentuh data cuaca hari t.
    """

    g = g.copy().sort_values("tanggal")

    # ============================================================
    # LAGGED SERIES (BASE UNTUK SEMUA FITUR)
    # ============================================================

    lagged = {}
    for col in WEATHER_COLS:
        short = SHORT_NAMES[col]
        lagged[col] = g[col].shift(1)
        g[f"lag1_{short}"] = lagged[col]

    # ============================================================
    # 1. WATER BALANCE (tetap)
    # ============================================================

    g["water_balance"] = (
        lagged["precipitation_sum"]
        - lagged["et0_fao_evapotranspiration"]
    )

    g["water_deficit"] = (
        lagged["et0_fao_evapotranspiration"]
        - lagged["precipitation_sum"]
    )

    rain = lagged["precipitation_sum"]
    et0  = lagged["et0_fao_evapotranspiration"]
    temp = lagged["temperature_2m_mean"]
    temp_min = lagged["temperature_2m_min"]
    vpd  = lagged["vapour_pressure_deficit_max"]
    rh   = lagged["relative_humidity_2m_mean"]
    wind = lagged["wind_speed_10m_max"]
    rad  = lagged["shortwave_radiation_sum"]
    soil = lagged["soil_moisture_0_to_7cm_mean"]

    # ============================================================
    # 2. EXISTING 3-DAY ROLLING FEATURES
    # ============================================================

    g["precipitation_3d_sum"] = rain.rolling(3, min_periods=1).sum()
    g["precipitation_3d_max"] = rain.rolling(3, min_periods=1).max()

    g["et0_3d_sum"] = et0.rolling(3, min_periods=1).sum()
    g["et0_3d_mean"] = et0.rolling(3, min_periods=1).mean()

    g["water_balance_3d"] = (
        g["precipitation_3d_sum"]
        - g["et0_3d_sum"]
    )

    g["temperature_3d_mean"] = temp.rolling(3, min_periods=1).mean()

    g["temperature_3d_min"] = temp_min.rolling(3, min_periods=1).min()

    temp_3d_max = temp.rolling(3, min_periods=1).max()

    g["temperature_3d_range"] = (
        temp_3d_max - g["temperature_3d_min"]
    )

    g["vpd_3d_max"] = vpd.rolling(3, min_periods=1).max()
    g["rh_3d_mean"] = rh.rolling(3, min_periods=1).mean()
    g["wind_3d_max"] = wind.rolling(3, min_periods=1).max()
    g["radiation_3d_sum"] = rad.rolling(3, min_periods=1).sum()
    g["soil_moisture_3d_mean"] = soil.rolling(3, min_periods=1).mean()

    # ============================================================
    # 3. EXISTING BINARY FLAGS
    # ============================================================

    g["is_heavy_rain_3d"] = (g["precipitation_3d_sum"] > 50).astype(int)
    g["is_dry_3d"] = (g["water_balance_3d"] < -30).astype(int)
    g["is_cold_spell_3d"] = (g["temperature_3d_min"] < 15).astype(int)
    g["is_vpd_extreme_3d"] = (g["vpd_3d_max"] > 2.0).astype(int)
    g["is_low_rh_3d"] = (g["rh_3d_mean"] < 60).astype(int)
    g["is_windy_3d"] = (g["wind_3d_max"] > 40).astype(int)

    # ============================================================
    # 4. EXISTING STRESS INDICES
    # ============================================================

    g["water_stress_3d"] = (
        g["et0_3d_sum"] / (g["precipitation_3d_sum"] + 1e-6)
    )

    g["heat_stress_3d"] = g["vpd_3d_max"]
    g["humidity_stress_3d"] = g["rh_3d_mean"]

    # ============================================================
    # 5. NEW: LONGER LAGS
    # ============================================================

    g["lag7_precipitation"] = g["precipitation_sum"].shift(7)
    g["lag7_temperature"]   = g["temperature_2m_mean"].shift(7)
    g["lag14_precipitation"] = g["precipitation_sum"].shift(14)

    # ============================================================
    # 6. NEW: CUMULATIVE RAINFALL
    # ============================================================

    g["rainfall_7d_sum"]  = rain.rolling(7, min_periods=1).sum()
    g["rainfall_14d_sum"] = rain.rolling(14, min_periods=1).sum()
    g["rainfall_30d_sum"] = rain.rolling(30, min_periods=1).sum()

    # ============================================================
    # 7. NEW: WEATHER ANOMALY
    # ============================================================

    temp_30d_mean = temp.rolling(30, min_periods=1).mean()
    g["temperature_anomaly"] = temp - temp_30d_mean

    rain_30d_mean = rain.rolling(30, min_periods=1).mean()
    g["precipitation_anomaly"] = rain - rain_30d_mean

    vpd_30d_mean = vpd.rolling(30, min_periods=1).mean()
    g["vpd_anomaly"] = vpd - vpd_30d_mean

    # ============================================================
    # 8. NEW: SEASON TRANSITION FLAGS
    # ============================================================

    # Deteksi transisi musim berdasarkan bulan dari kolom tanggal
    # (bukan dari lagged data — tanggal adalah informasi kalender, aman)
    month = g["tanggal"].dt.month

    # Transisi kemarau → hujan (Okt–Nov)
    # Tanaman sensitif pada perubahan dari kering ke basah
    g["is_musim_hujan_awal"] = (
        (month >= 10) & (month <= 11)
    ).astype(int)

    # Transisi hujan → kemarau (Apr–Jun)
    # Tanaman sensitif pada perubahan dari basah ke kering
    g["is_musim_kemarau_awal"] = (
        (month >= 4) & (month <= 6)
    ).astype(int)

    return g


def main():

    print(f"Memuat data: {INPUT_FILE}")
    df = load_data()
    print(
        f"  {len(df)} baris | "
        f"{df['kab_kota'].nunique()} kab/kota"
    )

    groups = []
    for nama, grp in df.groupby("kab_kota"):
        groups.append(feature_engineer_group(grp))

    result = pd.concat(groups, ignore_index=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSelesai!")
    print(f"Output: {OUTPUT_FILE}")

    print(f"\nJumlah kolom: {len(result.columns)}")

    # Tampilkan kolom baru
    new_cols = [
        "lag7_precipitation", "lag7_temperature", "lag14_precipitation",
        "rainfall_7d_sum", "rainfall_14d_sum", "rainfall_30d_sum",
        "temperature_anomaly", "precipitation_anomaly", "vpd_anomaly",
        "is_musim_hujan_awal", "is_musim_kemarau_awal",
    ]
    existing_new = [c for c in new_cols if c in result.columns]
    print(f"\nKolom baru ({len(existing_new)}):")
    for c in existing_new:
        print(f"  + {c}")

    print("\nPreview:")
    print(result.head())


if __name__ == "__main__":
    main()
