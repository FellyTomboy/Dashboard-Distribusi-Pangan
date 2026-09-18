import csv
import math
from datetime import date, timedelta

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

START_YEAR = 2018
END_YEAR   = 2025

HOLIDAY_FILE = "Data Lain-Lain/libur_nasional_2018-2025.csv"
OUTPUT_FILE  = f"data final/date_features_2018-2025.csv"

# ─────────────────────────────────────────────────────────────
# BACA DATA LIBUR
# ─────────────────────────────────────────────────────────────

holiday_dates: set[str] = set()
holiday_names: dict[str, str] = {}

with open(HOLIDAY_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        tanggal = row["tanggal"].strip()
        nama    = row["hari_raya"].strip()

        holiday_dates.add(tanggal)
        holiday_names[tanggal] = nama


# ─────────────────────────────────────────────────────────────
# TANGGAL-TANGGAL REFERENSI UNTUK PROXIMITY
# ─────────────────────────────────────────────────────────────

LEBARAN_DATES = [
    # 2018
    date(2018, 6, 15),
    # 2019
    date(2019, 6, 5),
    # 2020
    date(2020, 5, 24),
    # 2021
    date(2021, 5, 13),
    # 2022
    date(2022, 5, 2),
    # 2023
    date(2023, 4, 22),
    # 2024
    date(2024, 4, 10),
    # 2025
    date(2025, 3, 31),
]

NATAL_DATES = [
    date(2018, 12, 25),
    date(2019, 12, 25),
    date(2020, 12, 25),
    date(2021, 12, 25),
    date(2022, 12, 25),
    date(2023, 12, 25),
    date(2024, 12, 25),
    date(2025, 12, 25),
]

TAHUN_BARU_DATES = [
    date(2019, 1, 1),
    date(2020, 1, 1),
    date(2021, 1, 1),
    date(2022, 1, 1),
    date(2023, 1, 1),
    date(2024, 1, 1),
    date(2025, 1, 1),
    date(2026, 1, 1),
]


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def is_weekend(d: date) -> bool:
    """
    Sabtu=5, Minggu=6
    """
    return d.weekday() >= 5


def season(month: int) -> int:
    """
    1 = kemarau (Jun–Okt)
    0 = hujan   (Nov–Mei)
    """
    return 1 if 6 <= month <= 10 else 0


def is_lebaran_period(d: date) -> bool:
    """
    Deteksi periode Lebaran:
    ±7 hari dari Idul Fitri
    """
    for ld in LEBARAN_DATES:
        start = ld - timedelta(days=7)
        end   = ld + timedelta(days=7)
        if start <= d <= end:
            return True
    return False


def compute_week_flags(
    dates: list[date],
    week_of_year: list[int]
) -> tuple[list[int], list[int]]:
    """
    is_first_week_of_month:
        1 jika week tsb adalah minggu pertama
        yang menyentuh bulan tersebut

    is_last_week_of_month:
        1 jika week tsb adalah minggu terakhir
        yang menyentuh bulan tersebut
    """

    n = len(dates)

    first = [0] * n
    last  = [0] * n

    month_min_week: dict[tuple[int, int], int] = {}
    month_max_week: dict[tuple[int, int], int] = {}

    for i, d in enumerate(dates):

        key = (d.year, d.month)
        w   = week_of_year[i]

        if key not in month_min_week or w < month_min_week[key]:
            month_min_week[key] = w

        if key not in month_max_week or w > month_max_week[key]:
            month_max_week[key] = w

    for i, d in enumerate(dates):

        key = (d.year, d.month)
        w   = week_of_year[i]

        if w == month_min_week[key]:
            first[i] = 1

        if w == month_max_week[key]:
            last[i] = 1

    return first, last


def nearest_future(reference: date, targets: list[date]) -> int:
    """
    Hari ke depan menuju target terdekat (positif jika target di masa depan).
    Jika tidak ada target di masa depan, return 0.
    """
    min_days = 9999
    for t in targets:
        diff = (t - reference).days
        if 0 <= diff < min_days:
            min_days = diff
    return min_days if min_days < 9999 else 0


def nearest_past(reference: date, targets: list[date]) -> int:
    """
    Hari sejak target terdekat di masa lalu (positif).
    Jika tidak ada target di masa lalu, return 0.
    """
    min_days = 9999
    for t in targets:
        diff = (reference - t).days
        if 0 <= diff < min_days:
            min_days = diff
    return min_days if min_days < 9999 else 0


# ─────────────────────────────────────────────────────────────
# CYCLICAL ENCODING
# ─────────────────────────────────────────────────────────────

def cyclical_encode(value: float, period: float) -> tuple[float, float]:
    """Return (sin, cos) for cyclical encoding."""
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    start = date(START_YEAR, 1, 1)
    end   = date(END_YEAR, 12, 31)

    print("Generating date features...")
    print(f"Tahun : {START_YEAR}-{END_YEAR}")

    # ─────────────────────────────────────────────────────────
    # GENERATE SEMUA TANGGAL
    # ─────────────────────────────────────────────────────────

    all_dates: list[date] = []

    d = start

    while d <= end:

        all_dates.append(d)
        d += timedelta(days=1)

    print(f"Total tanggal: {len(all_dates)}")

    # ─────────────────────────────────────────────────────────
    # BASIC FEATURES
    # ─────────────────────────────────────────────────────────

    tanggal_str = [d.isoformat() for d in all_dates]

    day_of_week = [
        d.weekday()
        for d in all_dates
    ]

    month_col = [
        d.month
        for d in all_dates
    ]

    year_col = [
        d.year
        for d in all_dates
    ]

    day_col = [
        d.day
        for d in all_dates
    ]

    season_col = [
        season(d.month)
        for d in all_dates
    ]

    week_of_year = [
        d.isocalendar()[1]
        for d in all_dates
    ]

    quarter_col = [
        ((d.month - 1) // 3) + 1
        for d in all_dates
    ]

    # ─────────────────────────────────────────────────────────
    # CYCLICAL ENCODING (NEW)
    # ─────────────────────────────────────────────────────────

    month_sin = []
    month_cos = []

    week_sin = []
    week_cos = []

    dow_sin = []
    dow_cos = []

    for m, w, dow in zip(month_col, week_of_year, day_of_week):

        ms, mc = cyclical_encode(m, 12.0)
        month_sin.append(ms)
        month_cos.append(mc)

        ws, wc = cyclical_encode(w, 52.0)
        week_sin.append(ws)
        week_cos.append(wc)

        ds, dc = cyclical_encode(dow, 7.0)
        dow_sin.append(ds)
        dow_cos.append(dc)

    # ─────────────────────────────────────────────────────────
    # HOLIDAY FEATURES
    # ─────────────────────────────────────────────────────────

    is_holiday = []

    is_national_holiday = []

    is_cuti_bersama = []

    for d in all_dates:

        ds = d.isoformat()

        is_libur_nasional = (
            ds in holiday_dates
            and "cuti bersama" not in holiday_names[ds].lower()
        )

        is_cuti = (
            ds in holiday_dates
            and "cuti bersama" in holiday_names[ds].lower()
        )

        is_hol = (
            is_weekend(d)
            or ds in holiday_dates
        )

        is_holiday.append(
            1 if is_hol else 0
        )

        is_national_holiday.append(
            1 if is_libur_nasional else 0
        )

        is_cuti_bersama.append(
            1 if is_cuti else 0
        )

    # ─────────────────────────────────────────────────────────
    # LEBARAN FEATURE
    # ─────────────────────────────────────────────────────────

    is_lebaran_week = [
        1 if is_lebaran_period(d) else 0
        for d in all_dates
    ]

    # ─────────────────────────────────────────────────────────
    # NEW: HOLIDAY PROXIMITY
    # ─────────────────────────────────────────────────────────

    days_to_lebaran = [
        nearest_future(d, LEBARAN_DATES)
        for d in all_dates
    ]

    days_after_lebaran = [
        nearest_past(d, LEBARAN_DATES)
        for d in all_dates
    ]

    days_to_natal = [
        nearest_future(d, NATAL_DATES)
        for d in all_dates
    ]

    days_after_natal = [
        nearest_past(d, NATAL_DATES)
        for d in all_dates
    ]

    days_to_tahun_baru = [
        nearest_future(d, TAHUN_BARU_DATES)
        for d in all_dates
    ]

    days_after_tahun_baru = [
        nearest_past(d, TAHUN_BARU_DATES)
        for d in all_dates
    ]

    # ─────────────────────────────────────────────────────────
    # NEW: AWAL/ AKHIR BULAN (sudah ada week flags, tambah day-based)
    # ─────────────────────────────────────────────────────────

    is_early_month = [
        1 if 1 <= d.day <= 5 else 0
        for d in all_dates
    ]

    is_late_month = [
        1 if d.day >= 25 else 0
        for d in all_dates
    ]

    is_mid_month = [
        1 if 10 <= d.day <= 20 else 0
        for d in all_dates
    ]

    # ─────────────────────────────────────────────────────────
    # WEEK FEATURES
    # ─────────────────────────────────────────────────────────

    is_first_wom, is_last_wom = compute_week_flags(
        all_dates,
        week_of_year
    )

    # ─────────────────────────────────────────────────────────
    # WRITE CSV
    # ─────────────────────────────────────────────────────────

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "tanggal",
            "year",
            "month",
            "day",

            "day_of_week",
            "week_of_year",
            "quarter",

            # Cyclical encoding (NEW)
            "month_sin",
            "month_cos",
            "week_sin",
            "week_cos",
            "dow_sin",
            "dow_cos",

            "season",

            "is_holiday",
            "is_national_holiday",
            "is_cuti_bersama",

            "is_lebaran_week",

            # Holiday proximity (NEW)
            "days_to_lebaran",
            "days_after_lebaran",
            "days_to_natal",
            "days_after_natal",
            "days_to_tahun_baru",
            "days_after_tahun_baru",

            # Period flags (NEW)
            "is_early_month",
            "is_mid_month",
            "is_late_month",

            "is_first_week_of_month",
            "is_last_week_of_month",
        ])

        for i in range(len(all_dates)):

            writer.writerow([

                tanggal_str[i],

                year_col[i],
                month_col[i],
                day_col[i],

                day_of_week[i],
                week_of_year[i],
                quarter_col[i],

                # Cyclical encoding
                month_sin[i],
                month_cos[i],
                week_sin[i],
                week_cos[i],
                dow_sin[i],
                dow_cos[i],

                season_col[i],

                is_holiday[i],
                is_national_holiday[i],
                is_cuti_bersama[i],

                is_lebaran_week[i],

                # Holiday proximity
                days_to_lebaran[i],
                days_after_lebaran[i],
                days_to_natal[i],
                days_after_natal[i],
                days_to_tahun_baru[i],
                days_after_tahun_baru[i],

                # Period flags
                is_early_month[i],
                is_mid_month[i],
                is_late_month[i],

                is_first_wom[i],
                is_last_wom[i],
            ])

    # ─────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────

    print("\n===================================")
    print("SELESAI")
    print("===================================")

    print(f"Output : {OUTPUT_FILE}")
    print(f"Total  : {len(all_dates)} baris")
    print(f"Kolom  : 30 kolom (14 baru + 16 existing)")

    print("\nPreview 15 baris pertama:\n")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:

        for i, line in enumerate(f):

            if i > 15:
                break

            print(line.strip())


if __name__ == "__main__":
    main()
