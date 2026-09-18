#!/usr/bin/env python3
"""
Fetch OSRM Distance & Travel Time Matrix — East Java
======================================================
Mengambil jarak (meter) dan waktu tempuh (detik) dari OSRM untuk
semua pasangan kab/kota di Jawa Timur (38 wilayah).

Menggunakan OSRM demo server (router.project-osrm.org) atau
alternative: routing.openstreetmap.de.

Output: Data Lain-Lain/jarak_waktu_tempuh_osrm.csv
Columns: kab_asal, kab_tujuan, jarak_meter, jarak_km, waktu_detik, waktu_menit

DISCLAIMER: OSRM demo server memiliki rate limit.
Script ini menambahkan delay 1 detik antar request.
Untuk 38 wilayah = 703 pasangan -> ~12 menit.

Jika ingin lebih cepat: jalankan OSRM local atau pakai batch API.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from itertools import combinations
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================
KOORD_PATH = "Data Lain-Lain/koordinat_jatim.csv"
OUTPUT_PATH = "Data Lain-Lain/jarak_waktu_tempuh_osrm.csv"

# OSRM server: pilih salah satu
# OSRM_URL = "https://router.project-osrm.org"  # rate limit ketat
OSRM_URL = "https://routing.openstreetmap.de/routed-car"  # alternatif

# Delay antar request (detik) — untuk menghindari rate limit
REQUEST_DELAY = 1.0

# Format koordinat OSRM: lon,lat
def osrm_coord(lat, lon):
    return f"{lon},{lat}"

def fetch_osrm_route(lat1, lon1, lat2, lon2, retries=3):
    """
    Fetch jarak (meter) dan waktu (detik) dari OSRM untuk satu pasang koordinat.
    
    Returns (jarak_meter, waktu_detik) atau None jika gagal.
    """
    coord1 = osrm_coord(lat1, lon1)
    coord2 = osrm_coord(lat2, lon2)
    url = f"{OSRM_URL}/route/v1/driving/{coord1};{coord2}"
    params = {
        "overview": "false",
        "steps": "false",
        "annotations": "false",
    }
    
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data["code"] == "Ok" and len(data["routes"]) > 0:
                    route = data["routes"][0]
                    jarak_m = route["distance"]    # meter
                    waktu_s = route["duration"]    # detik
                    return jarak_m, waktu_s
                else:
                    print(f"       ⚠ OSRM tidak return route: {data.get('code', 'unknown')}")
            elif resp.status_code == 429:
                print(f"       ⚠ Rate limited (429), menunggu 5 detik...")
                time.sleep(5)
            else:
                print(f"       ⚠ HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            print(f"       ⚠ Timeout, percobaan {attempt+1}/{retries}")
        except requests.exceptions.ConnectionError:
            print(f"       ⚠ Connection error, percobaan {attempt+1}/{retries}")
        except Exception as e:
            print(f"       ⚠ Error: {e}")
        
        if attempt < retries - 1:
            time.sleep(2)
    
    return None


# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("  OSRM Distance Matrix — Jawa Timur (38 kab/kota)")
print("=" * 70)

# Load koordinat
df_koord = pd.read_csv(KOORD_PATH)
print(f"\n[1] Memuat {len(df_koord)} wilayah dari {KOORD_PATH}")
print(df_koord[['nama', 'latitude', 'longitude']].to_string())

# Semua pasangan
kab_list = df_koord['nama'].tolist()
n = len(kab_list)
total_pairs = n * (n - 1) // 2
print(f"\n[2] Total pasangan: {total_pairs}")

# Cek apakah file output sudah ada
if os.path.exists(OUTPUT_PATH):
    df_existing = pd.read_csv(OUTPUT_PATH)
    done_pairs = set(zip(df_existing['kab_asal'], df_existing['kab_tujuan']))
    print(f"\n     File output sudah ada: {len(df_existing)} pasangan tersimpan")
    print(f"     ({len(done_pairs)} / {total_pairs} pasangan unik)")
else:
    done_pairs = set()
    print(f"\n     File output belum ada, akan dibuat baru")

# Fetch
results = []
failed = 0
skipped = 0

print(f"\n[3] Mulai fetch dari OSRM ({OSRM_URL})...")
print(f"     Delay: {REQUEST_DELAY}s/request")
print(f"     Estimasi waktu: {total_pairs * REQUEST_DELAY / 60:.1f} menit (dengan delay)")
print()

for idx, (i, j) in enumerate(combinations(range(n), 2)):
    nama_i = kab_list[i]
    nama_j = kab_list[j]
    
    # Skip jika sudah ada
    if (nama_i, nama_j) in done_pairs or (nama_j, nama_i) in done_pairs:
        skipped += 1
        if skipped % 100 == 0:
            print(f"     [{idx+1}/{total_pairs}] {skipped} skipped (already exists)")
        continue
    
    lat_i = df_koord.iloc[i]['latitude']
    lon_i = df_koord.iloc[i]['longitude']
    lat_j = df_koord.iloc[j]['latitude']
    lon_j = df_koord.iloc[j]['longitude']
    
    print(f"     [{idx+1}/{total_pairs}] {nama_i[:25]:25s} <-> {nama_j[:25]:25s}", end=" ")
    
    result = fetch_osrm_route(lat_i, lon_i, lat_j, lon_j)
    
    if result:
        jarak_m, waktu_s = result
        jarak_km = round(jarak_m / 1000, 2)
        waktu_menit = round(waktu_s / 60, 1)
        print(f"✓ {jarak_km:.1f}km | {waktu_menit:.0f}menit")
        results.append({
            'kab_asal': nama_i,
            'kab_tujuan': nama_j,
            'jarak_meter': jarak_m,
            'jarak_km': jarak_km,
            'waktu_detik': waktu_s,
            'waktu_menit': waktu_menit,
        })
    else:
        print(f"✗ FAILED")
        failed += 1
        # Simpan NaN sebagai placeholder
        results.append({
            'kab_asal': nama_i,
            'kab_tujuan': nama_j,
            'jarak_meter': None,
            'jarak_km': None,
            'waktu_detik': None,
            'waktu_menit': None,
        })
    
    # Save incremental setiap 50 pasangan
    if (idx + 1) % 50 == 0 and results:
        df_temp = pd.DataFrame(results)
        if os.path.exists(OUTPUT_PATH):
            df_old = pd.read_csv(OUTPUT_PATH)
            df_temp = pd.concat([df_old, df_temp], ignore_index=True)
        df_temp.to_csv(OUTPUT_PATH, index=False)
        print(f"       → Auto-save: {len(df_temp)} total pasangan tersimpan")
        results = []  # reset buffer
    
    # Delay
    time.sleep(REQUEST_DELAY)

# Final save
if results:
    df_new = pd.DataFrame(results)
    if os.path.exists(OUTPUT_PATH):
        df_all = pd.concat([pd.read_csv(OUTPUT_PATH), df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(OUTPUT_PATH, index=False)
    print(f"\n     → Final save: {len(df_all)} pasangan tersimpan")
elif os.path.exists(OUTPUT_PATH):
    df_all = pd.read_csv(OUTPUT_PATH)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  RINGKASAN")
print("=" * 70)

if os.path.exists(OUTPUT_PATH):
    df_all = pd.read_csv(OUTPUT_PATH)
    print(f"\n  Total pasangan: {len(df_all)}")
    print(f"  Kolom: {list(df_all.columns)}")
    
    # Statistik
    valid = df_all.dropna(subset=['jarak_km'])
    print(f"  Berhasil: {len(valid)}")
    print(f"  Gagal: {df_all['jarak_km'].isna().sum()}")
    print(f"\n  Statistik jarak (valid):")
    print(f"     Mean  : {valid['jarak_km'].mean():.1f} km")
    print(f"     Median: {valid['jarak_km'].median():.1f} km")
    print(f"     Min   : {valid['jarak_km'].min():.1f} km")
    print(f"     Max   : {valid['jarak_km'].max():.1f} km")
    print(f"\n  Statistik waktu tempuh (valid):")
    print(f"     Mean  : {valid['waktu_menit'].mean():.0f} menit")
    print(f"     Median: {valid['waktu_menit'].median():.0f} menit")
    
    # Sample
    print(f"\n  5 pasangan terdekat:")
    nearest = valid.nsmallest(5, 'jarak_km')
    for _, r in nearest.iterrows():
        print(f"     {r['kab_asal'][:25]:25s} <-> {r['kab_tujuan'][:25]:25s} : {r['jarak_km']:.1f} km / {r['waktu_menit']:.0f} menit")
    
    print(f"\n  Output: {OUTPUT_PATH}")
else:
    print("\n  ❌ Tidak ada data tersimpan!")

print("\n  Selesai.")