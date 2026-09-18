import openmeteo_requests
import pandas as pd
import requests_cache

from retry_requests import retry

# ============================================================
# SETUP OPEN-METEO CLIENT
# ============================================================

cache_session = requests_cache.CachedSession(
    '.cache',
    expire_after=3600
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)

# ============================================================
# DATA KABUPATEN / KOTA
# ============================================================

wilayah_df = pd.DataFrame({
    "nama": [
        "Kabupaten Bangkalan",
        "Kabupaten Banyuwangi",
        "Kabupaten Blitar",
        "Kabupaten Bojonegoro",
        "Kabupaten Bondowoso",
        "Kabupaten Gresik",
        "Kabupaten Jember",
        "Kabupaten Jombang",
        "Kabupaten Kediri",
        "Kabupaten Lamongan",
        "Kabupaten Lumajang",
        "Kabupaten Madiun",
        "Kabupaten Magetan",
        "Kabupaten Malang",
        "Kabupaten Mojokerto",
        "Kabupaten Nganjuk",
        "Kabupaten Ngawi",
        "Kabupaten Pacitan",
        "Kabupaten Pamekasan",
        "Kabupaten Pasuruan",
        "Kabupaten Ponorogo",
        "Kabupaten Probolinggo",
        "Kabupaten Sampang",
        "Kabupaten Sidoarjo",
        "Kabupaten Situbondo",
        "Kabupaten Sumenep",
        "Kabupaten Trenggalek",
        "Kabupaten Tuban",
        "Kabupaten Tulungagung",
        "Kota Batu",
        "Kota Blitar",
        "Kota Kediri",
        "Kota Madiun",
        "Kota Malang",
        "Kota Mojokerto",
        "Kota Pasuruan",
        "Kota Probolinggo",
        "Kota Surabaya"
    ],

    "latitude": [
        -7.0482, -8.2192, -8.0933, -7.1507,
        -7.9097, -7.1571, -8.1724, -7.5481,
        -7.6833, -7.1175, -8.1308, -7.6667,
        -7.6476, -8.1300, -7.5377, -7.6046,
        -7.4067, -8.1979, -7.1572, -7.5956,
        -7.8651, -7.7572, -7.1819, -7.4479,
        -7.7063, -7.0167, -8.0534, -6.8982,
        -8.0650, -7.8710, -8.0953, -7.8156,
        -7.6298, -7.9797, -7.4714, -7.6456,
        -7.7541, -7.2575
    ],

    "longitude": [
        112.7465, 114.3691, 112.1767, 111.8817,
        113.8224, 112.6512, 113.7028, 112.2318,
        111.9667, 112.4151, 113.2227, 111.5833,
        111.3282, 112.5700, 112.5393, 111.9010,
        111.4467, 111.1042, 113.4724, 112.7877,
        111.4633, 113.4100, 113.2451, 112.7183,
        114.0076, 113.8667, 111.7101, 112.0490,
        111.9043, 112.5220, 112.1608, 112.0113,
        111.5239, 112.6304, 112.4344, 112.9073,
        113.2157, 112.7521
    ]
})

# ============================================================
# REQUEST API
# ============================================================

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": wilayah_df["latitude"].tolist(),

    "longitude": wilayah_df["longitude"].tolist(),

    "start_date": "2018-01-01",

    "end_date": "2021-12-31",

    "daily": [
        "temperature_2m_mean",
        "precipitation_sum",
        "et0_fao_evapotranspiration",
        "temperature_2m_min",
        "vapour_pressure_deficit_max",
        "relative_humidity_2m_mean",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "soil_moisture_0_to_7cm_mean"
    ]
}

responses = openmeteo.weather_api(
    url,
    params=params
)

# ============================================================
# PROCESS DATA
# ============================================================

all_data = []

for i, response in enumerate(responses):

    nama_wilayah = wilayah_df.iloc[i]["nama"]

    print(f"Processing: {nama_wilayah}")

    daily = response.Daily()

    daily_data = {
        "tanggal": pd.date_range(
            start=pd.to_datetime(
                daily.Time(),
                unit="s",
                utc=True
            ),
            end=pd.to_datetime(
                daily.TimeEnd(),
                unit="s",
                utc=True
            ),
            freq=pd.Timedelta(
                seconds=daily.Interval()
            ),
            inclusive="left"
        ),

        "kab_kota": nama_wilayah,

        "temperature_2m_mean":
            daily.Variables(0).ValuesAsNumpy(),

        "precipitation_sum":
            daily.Variables(1).ValuesAsNumpy(),

        "et0_fao_evapotranspiration":
            daily.Variables(2).ValuesAsNumpy(),

        "temperature_2m_min":
            daily.Variables(3).ValuesAsNumpy(),

        "vapour_pressure_deficit_max":
            daily.Variables(4).ValuesAsNumpy(),

        "relative_humidity_2m_mean":
            daily.Variables(5).ValuesAsNumpy(),

        "wind_speed_10m_max":
            daily.Variables(6).ValuesAsNumpy(),

        "shortwave_radiation_sum":
            daily.Variables(7).ValuesAsNumpy(),

        "soil_moisture_0_to_7cm_mean":
            daily.Variables(8).ValuesAsNumpy(),
    }

    df_daily = pd.DataFrame(daily_data)

    all_data.append(df_daily)

# ============================================================
# GABUNGKAN SEMUA DATA
# ============================================================

final_df = pd.concat(
    all_data,
    ignore_index=True
)

# ============================================================
# FORMAT TANGGAL
# ============================================================

final_df["tanggal"] = (
    final_df["tanggal"]
    .dt.tz_localize(None)
)

# ============================================================
# SORT DATA
# ============================================================

final_df.sort_values(
    ["kab_kota", "tanggal"],
    inplace=True
)

# ============================================================
# SIMPAN CSV
# ============================================================

output_file = "data_cuaca_jatim_2018_2021.csv"

final_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# OUTPUT INFO
# ============================================================

print("\nSelesai!")
print(f"Output tersimpan di: {output_file}")

print("\nJumlah data:")
print(final_df.shape)

print("\nPreview:")
print(final_df.head())