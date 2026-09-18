import csv
import urllib.request
import urllib.parse
import json
from datetime import datetime

API_URL = "https://api-hari-libur.vercel.app/api"


def fetch_holidays(year: int) -> list[tuple[str, str]]:
    """
    Fetch hari libur nasional + cuti bersama dari API Vercel.
    Return format:
    [
        ("2025-01-01", "Tahun Baru 2025 Masehi"),
        ...
    ]
    """

    params = urllib.parse.urlencode({"year": year})
    url = f"{API_URL}?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    rows = []

    # API return list langsung
    if isinstance(data, list):
        holidays = data

    # atau object {"data": [...]}
    elif isinstance(data, dict) and "data" in data:
        holidays = data["data"]

    else:
        raise RuntimeError(f"Format response API tidak dikenali untuk tahun {year}")

    for h in holidays:

        # fleksibel terhadap nama field API
        date = (
            h.get("date")
            or h.get("tanggal")
            or h.get("holiday_date")
            or ""
        )

        name = (
            h.get("name")
            or h.get("keterangan")
            or h.get("description")
            or ""
        )

        is_cuti = (
            h.get("is_cuti_bersama")
            or h.get("cuti")
            or "cuti bersama" in name.lower()
        )

        if not date or not name:
            continue

        if is_cuti and "cuti bersama" not in name.lower():
            name = f"Cuti Bersama {name}"

        rows.append((date, name))

    rows.sort(key=lambda x: x[0])
    return rows


def add_manual_2018_2019() -> list[tuple[str, str]]:
    """
    Tambahan manual data 2018-2019
    """

    manual = [

        # =========================
        # 2018
        # =========================

        ("2018-01-01", "Tahun Baru 2018 Masehi"),
        ("2018-02-16", "Tahun Baru Imlek 2569 Kongzili"),
        ("2018-03-17", "Hari Raya Nyepi Tahun Baru Saka 1940"),
        ("2018-03-30", "Wafat Isa Al Masih"),
        ("2018-04-14", "Isra Mikraj Nabi Muhammad SAW"),
        ("2018-05-01", "Hari Buruh Internasional"),
        ("2018-05-10", "Kenaikan Isa Al Masih"),
        ("2018-05-29", "Hari Raya Waisak 2562"),
        ("2018-06-01", "Hari Lahir Pancasila"),

        ("2018-06-15", "Hari Raya Idul Fitri 1439 Hijriyah"),
        ("2018-06-16", "Hari Raya Idul Fitri 1439 Hijriyah"),

        ("2018-08-17", "Hari Kemerdekaan Republik Indonesia"),
        ("2018-08-22", "Hari Raya Idul Adha 1439 Hijriyah"),
        ("2018-09-11", "Tahun Baru Islam 1440 Hijriyah"),
        ("2018-11-20", "Maulid Nabi Muhammad SAW"),
        ("2018-12-25", "Hari Raya Natal"),

        # cuti bersama 2018
        ("2018-06-13", "Cuti Bersama Hari Raya Idul Fitri 1439 Hijriyah"),
        ("2018-06-14", "Cuti Bersama Hari Raya Idul Fitri 1439 Hijriyah"),
        ("2018-06-18", "Cuti Bersama Hari Raya Idul Fitri 1439 Hijriyah"),
        ("2018-06-19", "Cuti Bersama Hari Raya Idul Fitri 1439 Hijriyah"),
        ("2018-12-24", "Cuti Bersama Hari Raya Natal"),

        # =========================
        # 2019
        # =========================

        ("2019-01-01", "Tahun Baru 2019 Masehi"),
        ("2019-02-05", "Tahun Baru Imlek 2570 Kongzili"),
        ("2019-03-07", "Hari Raya Nyepi Tahun Baru Saka 1941"),
        ("2019-04-03", "Isra Mikraj Nabi Muhammad SAW"),
        ("2019-04-19", "Wafat Isa Al Masih"),
        ("2019-05-01", "Hari Buruh Internasional"),
        ("2019-05-19", "Hari Raya Waisak 2563"),
        ("2019-05-30", "Kenaikan Isa Al Masih"),
        ("2019-06-01", "Hari Lahir Pancasila"),

        # cuti bersama idul fitri
        ("2019-06-03", "Cuti Bersama Hari Raya Idul Fitri 1440 Hijriyah"),
        ("2019-06-04", "Cuti Bersama Hari Raya Idul Fitri 1440 Hijriyah"),

        # idul fitri
        ("2019-06-05", "Hari Raya Idul Fitri 1440 Hijriyah"),
        ("2019-06-06", "Hari Raya Idul Fitri 1440 Hijriyah"),

        # cuti bersama lagi
        ("2019-06-07", "Cuti Bersama Hari Raya Idul Fitri 1440 Hijriyah"),

        ("2019-08-11", "Hari Raya Idul Adha 1440 Hijriyah"),
        ("2019-08-17", "Hari Kemerdekaan Republik Indonesia"),
        ("2019-09-01", "Tahun Baru Islam 1441 Hijriyah"),
        ("2019-11-09", "Maulid Nabi Muhammad SAW"),

        # natal
        ("2019-12-24", "Cuti Bersama Hari Raya Natal"),
        ("2019-12-25", "Hari Raya Natal"),
    ]

    return manual


def main():

    print("Mengambil data hari libur Indonesia...")
    print(">> Sumber API : api-hari-libur.vercel.app")
    print(">> Tahun API  : 2020-2025")
    print(">> Tahun manual : 2018-2019")

    all_rows = []

    # manual 2018-2019
    all_rows.extend(add_manual_2018_2019())

    # fetch API 2020-2025
    for year in range(2020, 2026):

        print(f"\nFetching {year}...")

        try:
            rows = fetch_holidays(year)

            print(f"  Berhasil: {len(rows)} data")

            all_rows.extend(rows)

        except Exception as e:
            print(f"  ERROR {year}: {e}")

    # sort final
    all_rows.sort(key=lambda x: x[0])

    # remove duplicate
    unique_rows = []
    seen = set()

    for row in all_rows:
        if row not in seen:
            unique_rows.append(row)
            seen.add(row)

    nasional = [
        r for r in unique_rows
        if "cuti bersama" not in r[1].lower()
    ]

    cuti = [
        r for r in unique_rows
        if "cuti bersama" in r[1].lower()
    ]

    output_file = "Data Lain-Lain/libur_nasional_2018-2025.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "tanggal",
            "hari_raya"
        ])

        writer.writerows(unique_rows)

    print("\n==============================")
    print("SELESAI")
    print("==============================")

    print(f"Total data             : {len(unique_rows)}")
    print(f"Hari Libur Nasional    : {len(nasional)}")
    print(f"Cuti Bersama           : {len(cuti)}")

    print(f"\nOutput:")
    print(f"  {output_file}")

    print("\nPreview data:")

    for t, k in unique_rows[:20]:

        tanda = (
            "[CUTI]"
            if "cuti bersama" in k.lower()
            else "[NAS]"
        )

        print(f"  {t} {tanda} {k}")


if __name__ == "__main__":
    main()