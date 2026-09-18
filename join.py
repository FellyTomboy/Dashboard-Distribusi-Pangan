import pandas as pd

# ==========================================
# 1. Mapping location_id -> nama kab/kota
# ==========================================

kabkota_mapping = {
    0: "Kabupaten Bangkalan",
    1: "Kabupaten Banyuwangi",
    2: "Kabupaten Blitar",
    3: "Kabupaten Bojonegoro",
    4: "Kabupaten Bondowoso",
    5: "Kabupaten Gresik",
    6: "Kabupaten Jember",
    7: "Kabupaten Jombang",
    8: "Kabupaten Kediri",
    9: "Kabupaten Lamongan",
    10: "Kabupaten Lumajang",
    11: "Kabupaten Madiun",
    12: "Kabupaten Magetan",
    13: "Kabupaten Malang",
    14: "Kabupaten Mojokerto",
    15: "Kabupaten Nganjuk",
    16: "Kabupaten Ngawi",
    17: "Kabupaten Pacitan",
    18: "Kabupaten Pamekasan",
    19: "Kabupaten Pasuruan",
    20: "Kabupaten Ponorogo",
    21: "Kabupaten Probolinggo",
    22: "Kabupaten Sampang",
    23: "Kabupaten Sidoarjo",
    24: "Kabupaten Situbondo",
    25: "Kabupaten Sumenep",
    26: "Kabupaten Trenggalek",
    27: "Kabupaten Tuban",
    28: "Kabupaten Tulungagung",
    29: "Kota Batu",
    30: "Kota Blitar",
    31: "Kota Kediri",
    32: "Kota Madiun",
    33: "Kota Malang",
    34: "Kota Mojokerto",
    35: "Kota Pasuruan",
    36: "Kota Probolinggo",
    37: "Kota Surabaya"
}

# ==========================================
# 2. Load data cuaca
# ==========================================

df_weather = pd.read_csv("Data Lain-Lain/open-meteo.csv", encoding="utf-8")

# ==========================================
# 3. Tambahkan nama kab/kota
# ==========================================

df_weather["kab_kota"] = df_weather["location_id"].map(kabkota_mapping)

# ==========================================
# 4. Rapikan kolom
# ==========================================

df_weather = df_weather[
    [
        "kab_kota",
        "time",
        "temperature_2m_mean (°C)",
        "precipitation_sum (mm)",
        "et0_fao_evapotranspiration (mm)",
        "temperature_2m_min (°C)",
        "vapour_pressure_deficit_max (kPa)",
        "relative_humidity_2m_mean (%)",
        "wind_speed_10m_max (km/h)",
        "shortwave_radiation_sum (MJ/m²)",
        "soil_moisture_0_to_7cm_mean (m³/m³)"
    ]
]

df_weather.to_csv("weather_with_kabkota.csv", index=False)

print(df_weather.head())