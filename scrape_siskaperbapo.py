"""
Scraper SISKAPERBAPO — Universal scraper for all commodities.
SELF-CONTAINED — bisa jalan di Windows tanpa perlu realtime.config.

Usage di Windows:
    python scrape_siskaperbapo.py                          # Full history semua komoditas
    python scrape_siskaperbapo.py --daily                   # 7 hari terakhir

Konfigurasi komoditas di KOMODITAS_LIST bawah.
Tambah komoditas baru cukup tambah satu dict di KOMODITAS_LIST.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date, timedelta
import time
import re
import logging
import sys
from pathlib import Path

# ── KOMODITAS LIST (Single Source of Truth) ────────────────────────────────
# Copy dari realtime/config.py. Jika ada perubahan, update KOMODITAS_LIST
# di BEFA file ini DAN di realtime/config.py.
KOMODITAS_LIST = [
    {
        "nama": "Bawang Merah",
        "siskaperbapo_id": "39",
    },
    {
        "nama": "Cabai Rawit Merah",
        "siskaperbapo_id": "50",
    },
]

# ── Konfigurasi ──────────────────────────────────────────────────────────────
BASE_URL      = "https://siskaperbapo.jatimprov.go.id"
ENDPOINT      = f"{BASE_URL}/harga-komoditas"
WINDOW_DAYS   = 7
DELAY_SEC     = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scrape_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": ENDPOINT,
    "Origin":  BASE_URL,
})


def get_csrf_token() -> str:
    """Ambil CSRF token dari halaman GET."""
    resp = SESSION.get(ENDPOINT, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    el = soup.find("input", {"name": "csrf_token"})
    if not el:
        raise ValueError("CSRF token tidak ditemukan!")
    return el["value"]


def fetch_weekly_data(end_date: date, csrf_token: str, komoditas_id: str) -> dict:
    """
    POST request untuk tanggal_akhir dan komoditas tertentu.
    Returns: {tanggal_str: {kab_kota: rata_rata_harga}}
    """
    payload = {
        "csrf_token":    csrf_token,
        "tanggal_akhir": end_date.strftime("%Y-%m-%d"),
        "komoditas":     komoditas_id,
    }
    resp = SESSION.post(ENDPOINT, data=payload, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if not table:
        log.warning(f"  Tidak ada tabel untuk end_date={end_date}")
        return {}

    header_row = table.find("tr")
    headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
    if len(headers) < 3:
        return {}
    date_cols = headers[2:]

    all_rows = table.find_all("tr")[1:]
    data: dict[str, dict[str, list]] = {}
    current_kab = None

    for row in all_rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        no_cell = cells[0]
        nama_cell = cells[1]
        harga_cells = cells[2:]

        is_bold = "bold" in (no_cell.get("style", "") or "")
        no_text = no_cell.get_text(strip=True)
        if is_bold and re.match(r"^\d+$", no_text):
            current_kab = nama_cell.get_text(strip=True)
            data[current_kab] = {t: [] for t in date_cols}
            continue
        if current_kab is None:
            continue
        for i, hcell in enumerate(harga_cells):
            if i >= len(date_cols):
                break
            tgl = date_cols[i]
            raw = hcell.get_text(strip=True).replace(".", "").replace(",", ".")
            if raw in ("-", ""):
                continue
            try:
                h = float(raw)
                if h > 0:
                    data[current_kab][tgl].append(h)
            except ValueError:
                pass

    result = {}
    for tgl in date_cols:
        result[tgl] = {}
        for kab, tgl_map in data.items():
            hl = tgl_map.get(tgl, [])
            result[tgl][kab] = round(sum(hl) / len(hl), 2) if hl else None
    return result


def generate_end_dates(start: date, end: date, window: int) -> list[date]:
    """Generate end_date untuk scraping window hari."""
    dates = []
    current_end = start + timedelta(days=window - 1)
    while current_end <= end:
        dates.append(current_end)
        current_end += timedelta(days=window)
    if dates and dates[-1] < end:
        dates.append(end)
    return dates


def scrape_commodity(
    komoditas_id: str,
    komoditas_nama: str,
    start_date: date = date(2024, 1, 1),
    end_date: date = date.today(),
) -> pd.DataFrame:
    """
    Scrape data harga untuk satu komoditas dalam rentang tanggal.
    Returns: DataFrame [tanggal, kab_kota, rata_harga]
    """
    log.info(f"\n{'='*60}")
    log.info(f"Scraping: {komoditas_nama} (ID: {komoditas_id})")
    log.info(f"Periode: {start_date} s/d {end_date}")
    log.info(f"{'='*60}")

    end_dates = generate_end_dates(start_date, end_date, WINDOW_DAYS)
    log.info(f"Total request: {len(end_dates)}")

    all_data: dict[str, dict[str, float]] = {}

    for i, ed in enumerate(end_dates):
        log.info(f"[{i+1}/{len(end_dates)}] end_date={ed} ...")
        for attempt in range(3):
            try:
                csrf = get_csrf_token()
                time.sleep(0.5)
                weekly = fetch_weekly_data(ed, csrf, komoditas_id)
                break
            except Exception as e:
                log.warning(f"  Attempt {attempt+1} gagal: {e}")
                if attempt == 2:
                    log.error(f"  Skip end_date={ed}")
                    weekly = {}
                time.sleep(5)

        for tgl_str, kab_map in weekly.items():
            try:
                tgl = date.fromisoformat(tgl_str)
            except ValueError:
                continue
            if start_date <= tgl <= end_date:
                if tgl_str not in all_data:
                    all_data[tgl_str] = kab_map
                else:
                    all_data[tgl_str].update({
                        k: v for k, v in kab_map.items()
                        if v is not None and all_data[tgl_str].get(k) is None
                    })

        log.info(f"  → {len(weekly)} tanggal")
        time.sleep(DELAY_SEC)

    if not all_data:
        log.error(f"Tidak ada data untuk {komoditas_nama}!")
        return pd.DataFrame()

    records = []
    for tgl_str in sorted(all_data.keys()):
        for kab, harga in all_data[tgl_str].items():
            if harga is not None and harga > 0:
                records.append({"tanggal": tgl_str, "kab_kota": kab, "rata_harga": harga})

    df = pd.DataFrame(records)
    log.info(f"✅ {komoditas_nama}: {len(df)} baris ({start_date} s/d {end_date})")
    return df


def scrape_recent(komoditas_id: str, days: int = 7) -> pd.DataFrame:
    """Scrape N hari terakhir untuk komoditas tertentu."""
    end = date.today()
    start = end - timedelta(days=days)
    return scrape_commodity(komoditas_id, f"Recent {days}d", start, end)


def main():
    """
    Scrape full history semua komoditas. Output CSV di folder yang sama.
    """
    start_date = date(2024, 1, 1)
    end_date = date.today()

    for kom in KOMODITAS_LIST:
        nama = kom["nama"]
        kom_id = kom["siskaperbapo_id"]
        df = scrape_commodity(kom_id, nama, start_date, end_date)
        
        if not df.empty:
            csv_name = f"prices_{nama.replace(' ', '_')}.csv"
            df.to_csv(csv_name, index=False, encoding="utf-8-sig")
            log.info(f"  💾 {csv_name} ({len(df)} baris)")

    log.info("\n✅ SEMUA SELESAI!")


if __name__ == "__main__":
    # --daily: hanya 7 hari terakhir (update cepat)
    if "--daily" in sys.argv:
        for kom in KOMODITAS_LIST:
            df = scrape_recent(kom["siskaperbapo_id"], days=7)
            if not df.empty:
                csv_name = f"prices_{kom['nama'].replace(' ', '_')}.csv"
                df.to_csv(csv_name, index=False)
                print(f"  💾 {csv_name}")
    else:
        main()