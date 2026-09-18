import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from pandas.api.types import (
    is_numeric_dtype,
)

from joblib import Parallel, delayed, dump
import time

# =========================================================
# CONFIG
# =========================================================

DATA_PATH = "cabai rawit\\training\\data_model_cabai_rawit.csv"

TARGET_COL = "rata_harga"
DATE_COL = "tanggal"
GROUP_COL = "kab_kota"

FORECAST_HORIZON = 7
RANDOM_STATE = 42

MODEL_SAVE_PATH = "cabai rawit\\training\\model_cabai_rawit.pkl"

# =========================================================
# LOAD DATA
# =========================================================

print("=" * 80)
print("LOAD DATA")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])

df = df.sort_values(
    [GROUP_COL, DATE_COL]
).reset_index(drop=True)

print(f"Jumlah data : {len(df):,}")
print(f"Jumlah fitur: {len(df.columns)}")

# =========================================================
# SPLIT DATA
# =========================================================

print("\n" + "=" * 80)
print("SPLIT DATA")
print("=" * 80)

train_df = df[
    (df[DATE_COL].dt.year >= 2018) &
    (df[DATE_COL].dt.year <= 2022)
].copy()

val_df = df[
    df[DATE_COL].dt.year == 2023
].copy()

test_df = df[
    (df[DATE_COL].dt.year >= 2024) &
    (df[DATE_COL].dt.year <= 2025)
].copy()

print(
    f"Train      : "
    f"{train_df[DATE_COL].min()} -> "
    f"{train_df[DATE_COL].max()} | "
    f"{len(train_df):,} rows"
)

print(
    f"Validation : "
    f"{val_df[DATE_COL].min()} -> "
    f"{val_df[DATE_COL].max()} | "
    f"{len(val_df):,} rows"
)

print(
    f"Test       : "
    f"{test_df[DATE_COL].min()} -> "
    f"{test_df[DATE_COL].max()} | "
    f"{len(test_df):,} rows"
)

# =========================================================
# FEATURE LIST
# =========================================================

EXCLUDE_COLS = [
    TARGET_COL,
    DATE_COL,

    # Fitur spasial volatil — tidak di-update per horizon recursive
    # (lihat buktistabilitasspasial.py: RelChange H7 > 30%, AutoCorr lag7 < 0.6)
    "deviasi_rata_harga",
    "pct_diff_harga_provinsi",
    "rank_harga_kab",
    "selisih_harga_tetangga",
]

FEATURE_COLS = [
    c for c in df.columns
    if c not in EXCLUDE_COLS
]

print(f"\nJumlah feature digunakan: {len(FEATURE_COLS)}")

# =========================================================
# HANDLE MISSING VALUE
# =========================================================

print("\n" + "=" * 80)
print("HANDLE MISSING VALUE")
print("=" * 80)

NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = []

for col in FEATURE_COLS:

    # numeric
    if is_numeric_dtype(train_df[col]):

        NUMERIC_FEATURES.append(col)

        median_value = train_df[col].median()

        train_df[col] = train_df[col].fillna(median_value)
        val_df[col] = val_df[col].fillna(median_value)
        test_df[col] = test_df[col].fillna(median_value)

    # categorical/string
    else:

        CATEGORICAL_FEATURES.append(col)

        train_df[col] = (
            train_df[col]
            .astype(str)
            .fillna("unknown")
            .astype("category")
        )

        val_df[col] = (
            val_df[col]
            .astype(str)
            .fillna("unknown")
            .astype("category")
        )

        test_df[col] = (
            test_df[col]
            .astype(str)
            .fillna("unknown")
            .astype("category")
        )

print(f"Numeric features     : {len(NUMERIC_FEATURES)}")
print(f"Categorical features : {CATEGORICAL_FEATURES}")

# =========================================================
# CACHE MEDIAN VALUES (untuk safe_fill_features)
# =========================================================

print("\n" + "=" * 80)
print("CACHE MEDIAN VALUES")
print("=" * 80)

MEDIAN_CACHE = {
    col: train_df[col].median()
    for col in NUMERIC_FEATURES
}

print(f"Median cached untuk {len(MEDIAN_CACHE)} fitur numerik.")

# =========================================================
# TRAIN MODEL
# =========================================================

print("\n" + "=" * 80)
print("TRAIN MODEL")
print("=" * 80)

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]

X_val = val_df[FEATURE_COLS]
y_val = val_df[TARGET_COL]

model = LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",

    n_estimators=1000,
    learning_rate=0.03,

    max_depth=8,
    num_leaves=63,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.1,
    reg_lambda=1.0,

    min_child_samples=20,

    random_state=RANDOM_STATE,
    n_jobs=-1,

    verbosity=-1,
    force_col_wise=True,
)

model.fit(
    X_train,
    y_train,

    eval_set=[(X_val, y_val)],
    eval_metric="l1",

    categorical_feature=CATEGORICAL_FEATURES,
)

print("Training selesai.")

# =========================================================
# SAVE MODEL
# =========================================================

print("\n" + "=" * 80)
print("SAVE MODEL")
print("=" * 80)

dump(model, MODEL_SAVE_PATH)
print(f"Model saved → {MODEL_SAVE_PATH}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE — stopping here.")
print("=" * 80)

# =========================================================
# CODE DI BAWAH INI TIDAK DIEKSEKUSI (DISIMPAN SEBAGAI REFERENSI)
# Semua kode mulai dari recursive forecast, evaluasi, penyimpanan prediksi,
# dan feature importance sengaja dikomentari agar proses training berhenti
# setelah model selesai disimpan.
# =========================================================

"""
# =========================================================
# COMPUTE PRICE FEATURES FROM ARRAY (optimized)
# =========================================================

# SEMUA kolom yang merupakan turunan harga (harus di-recompute saat rekursi)
PRICE_FEATURE_COLS = [
    # Lags
    "lag1_rata_harga", "lag2_rata_harga", "lag3_rata_harga",
    "lag7_rata_harga", "lag14_rata_harga", "lag21_rata_harga", "lag30_rata_harga",
    # Log transform
    "log_lag1_rata_harga",
    # Delta & pct
    "delta_harga", "pct_change_harga",
    # Rolling mean
    "harga_3d_mean", "harga_7d_mean", "harga_14d_mean", "harga_30d_mean",
    # Rolling min/max
    "harga_3d_min", "harga_3d_max",
    "harga_7d_min", "harga_7d_max",
    # Rolling std / volatilitas
    "harga_7d_std", "volatilitas_3d",
    # Trend / momentum
    "harga_trend_3d", "is_price_up",
    "momentum_vs_7d", "momentum_7d_vs_30d",
    "trend_7d", "trend_14d",
    # Z-score
    "zscore_harga_7d",
]


def compute_price_features_from_array(prices):
    \"\"\"
    Compute ALL price-derived features from a numpy array of prices
    (oldest first, newest last).

    Mencocokkan logika di feature engineering/data_harga.py:
    - lag1 = shift(1) = prices[-1]
    - rolling dihitung dari lag1_price (hanya masa lalu)
    \"\"\"
    n = len(prices)
    out = {}

    # -------------------------------------------------
    # LAG features
    # prices[-1] = harga terakhir yang diketahui (shift1 / lag1)
    # -------------------------------------------------
    lag1 = prices[-1] if n >= 1 else np.nan
    lag2 = prices[-2] if n >= 2 else np.nan
    lag3 = prices[-3] if n >= 3 else np.nan

    out["lag1_rata_harga"] = lag1
    out["lag2_rata_harga"] = lag2
    out["lag3_rata_harga"] = lag3
    out["lag7_rata_harga"] = prices[-7] if n >= 7 else np.nan
    out["lag14_rata_harga"] = prices[-14] if n >= 14 else np.nan
    out["lag21_rata_harga"] = prices[-21] if n >= 21 else np.nan
    out["lag30_rata_harga"] = prices[-30] if n >= 30 else np.nan

    # -------------------------------------------------
    # LOG TRANSFORM (dari lag1)
    # -------------------------------------------------
    out["log_lag1_rata_harga"] = np.log(lag1 + 1) if n >= 1 else np.nan

    # -------------------------------------------------
    # ROLLING — semua dari lag1_price (hanya masa lalu)
    # Sesuai data_harga.py: rolling dari lag1_price, bukan raw price
    # -------------------------------------------------
    # Hanya price sebelum current (history_prices = semua yg diketahui)
    # sudah benar karena current_actual_price tidak dimasukkan

    harga_3d = prices[-3:] if n >= 3 else prices
    harga_7d = prices[-7:] if n >= 7 else prices
    harga_14d = prices[-14:] if n >= 14 else prices
    harga_30d = prices[-30:] if n >= 30 else prices

    out["harga_3d_mean"] = np.mean(harga_3d) if n >= 3 else np.nan
    out["harga_3d_min"] = np.min(harga_3d) if n >= 3 else np.nan
    out["harga_3d_max"] = np.max(harga_3d) if n >= 3 else np.nan
    out["volatilitas_3d"] = np.std(harga_3d, ddof=1) if n >= 3 else (np.nan if n < 2 else np.std(prices, ddof=1))

    out["harga_7d_mean"] = np.mean(harga_7d) if n >= 7 else (np.mean(prices) if n > 0 else np.nan)
    out["harga_7d_min"] = np.min(harga_7d) if n >= 7 else (np.min(prices) if n > 0 else np.nan)
    out["harga_7d_max"] = np.max(harga_7d) if n >= 7 else (np.max(prices) if n > 0 else np.nan)
    out["harga_7d_std"] = (
        np.std(harga_7d, ddof=1) if n >= 8
        else (np.std(prices, ddof=1) if n >= 2 else np.nan)
    )

    out["harga_14d_mean"] = np.mean(harga_14d) if n >= 14 else (np.mean(prices) if n > 0 else np.nan)
    out["harga_30d_mean"] = np.mean(harga_30d) if n >= 30 else (np.mean(prices) if n > 0 else np.nan)

    # -------------------------------------------------
    # DELTA & PCT_CHANGE
    # Sesuai data_harga.py: delta = lag1 - lag2
    # -------------------------------------------------
    out["delta_harga"] = lag1 - lag2 if n >= 2 else np.nan
    out["pct_change_harga"] = (
        (lag1 - lag2) / (lag2 + 1e-6)
        if n >= 2 else np.nan
    )

    # -------------------------------------------------
    # TREND / MOMENTUM
    # -------------------------------------------------
    harga_3d_mean = out["harga_3d_mean"]
    harga_7d_mean = out["harga_7d_mean"]
    harga_14d_mean = out["harga_14d_mean"]
    harga_30d_mean = out["harga_30d_mean"]
    harga_7d_std = out["harga_7d_std"]

    out["harga_trend_3d"] = lag1 - harga_3d_mean if n >= 3 else np.nan
    out["is_price_up"] = 1 if (n >= 2 and out["delta_harga"] > 0) else (0 if n >= 2 else np.nan)
    out["momentum_vs_7d"] = lag1 - harga_7d_mean if n >= 7 else np.nan
    out["momentum_7d_vs_30d"] = harga_7d_mean - harga_30d_mean if n >= 30 else np.nan
    out["trend_7d"] = harga_7d_mean - harga_14d_mean if n >= 14 else np.nan
    out["trend_14d"] = harga_14d_mean - harga_30d_mean if n >= 30 else np.nan

    # -------------------------------------------------
    # Z-SCORE
    # Sesuai data_harga.py: rolling_mean & std dengan min_periods=3
    # -------------------------------------------------
    if n >= 7 and np.isfinite(harga_7d_std) and harga_7d_std != 0:
        out["zscore_harga_7d"] = (lag1 - harga_7d_mean) / (harga_7d_std + 1e-6)
    elif n >= 3:
        local_mean = np.mean(prices[-3:])
        local_std = np.std(prices[-3:], ddof=1)
        out["zscore_harga_7d"] = (lag1 - local_mean) / (local_std + 1e-6) if np.isfinite(local_std) else 0.0
    else:
        out["zscore_harga_7d"] = 0.0

    return out


# =========================================================
# SAFE FILL FUNCTION (dengan MEDIAN_CACHE)
# =========================================================

def safe_fill_features(row_df):
    \"\"\"
    Isi NaN menggunakan median yang sudah di-cache.
    Tidak perlu hitung train_df[col].median() setiap kali.
    \"\"\"
    row_df = row_df.copy()

    for col in NUMERIC_FEATURES:
        if col in row_df.columns:
            val = row_df[col].iloc[0]
            if pd.isna(val):
                row_df[col] = MEDIAN_CACHE[col]

    for col in CATEGORICAL_FEATURES:
        if col in row_df.columns:
            row_df[col] = (
                row_df[col]
                .astype(str)
                .fillna("unknown")
                .astype("category")
            )

    return row_df


# =========================================================
# PROCESS SINGLE GROUP (untuk parallel processing)
# =========================================================

def process_group(kab_kota, kab_future_df, history_df, horizon):
    \"\"\"
    Process recursive forecast untuk satu kab_kota.
    Mengembalikan list of dict predictions.
    \"\"\"
    predictions = []

    # --- history untuk group ini (hanya data training) ---
    kab_history = history_df[
        history_df[GROUP_COL] == kab_kota
    ].sort_values(DATE_COL)
    base_history = kab_history[TARGET_COL].values.astype(np.float64)

    # --- future data untuk group ini (di-sort & di-index) ---
    kab_future_sorted = (
        kab_future_df
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    unique_dates = kab_future_sorted[DATE_COL].unique()
    max_history = 200

    # Map tanggal ke posisi index untuk aksep cepat
    date_to_idx = {
        pd.Timestamp(d): i
        for i, d in enumerate(kab_future_sorted[DATE_COL].values)
    }

    # --- WALK FORWARD ---
    for current_date in unique_dates:
        current_date = pd.Timestamp(current_date)
        start_idx = date_to_idx[current_date]

        # ambil horizon baris dari current_date
        future_rows = kab_future_sorted.iloc[start_idx:start_idx + horizon]

        if len(future_rows) < horizon:
            continue

        # ============================================================
        # FIX: Gabungkan training history + actual prices SEBELUM current_date
        # Sehingga lag1 selalu merupakan harga aktual yang paling terkini
        # ============================================================
        if start_idx > 0:
            past_actual = kab_future_sorted.iloc[:start_idx][TARGET_COL].values
            history_prices = np.concatenate([
                base_history, past_actual
            ]).astype(np.float64)
        else:
            history_prices = base_history.copy()

        if len(history_prices) > max_history:
            history_prices = history_prices[-max_history:]

        # --- H+1 -> H+7 ---
        for step in range(horizon):
            # ambil 1 baris future (pakai index absolut)
            future_idx = start_idx + step
            current_row = kab_future_sorted.iloc[future_idx:future_idx + 1].copy()
            current_actual_price = kab_future_sorted.iloc[future_idx][TARGET_COL]

            # --- hitung price features DARI HISTORY SAJA (tanpa actual future) ---
            # history_prices = training + actual prices s.d. (current_date - 1)
            #                + prediksi recursive sebelumnya (jika step > 0)
            price_feats = compute_price_features_from_array(history_prices)

            # update SEMUA kolom price features di current_row
            for col, val in price_feats.items():
                if col in current_row.columns:
                    current_row[col] = val

            # --- safe fill ---
            current_row = safe_fill_features(current_row)

            # --- predict ---
            X_pred = current_row[FEATURE_COLS]
            pred_price = model.predict(X_pred)[0]

            predictions.append({
                "kab_kota": kab_kota,
                "forecast_date": kab_future_sorted.iloc[future_idx][DATE_COL],
                "horizon": step + 1,
                "actual": current_actual_price,
                "prediction": pred_price,
            })

            # --- append PREDICTED price ke history untuk step berikutnya ---
            history_prices = np.append(history_prices, pred_price)

            if len(history_prices) > max_history:
                history_prices = history_prices[-max_history:]

    return predictions


# =========================================================
# RECURSIVE FORECAST (parallel version)
# =========================================================

def recursive_forecast(
    model,
    history_df,
    future_df,
    horizon=7,
    n_jobs=-1,
    forecast_label="",
):
    \"\"\"
    Recursive forecast dengan parallel processing per group.

    Parameters
    ----------
    model : LGBMRegressor
    history_df : DataFrame
    future_df : DataFrame
    horizon : int
    n_jobs : int
        Jumlah CPU core (-1 = all cores)
    forecast_label : str
        Label untuk logging (e.g. "VALIDATION" / "TEST")
    \"\"\"
    grouped_future = future_df.groupby(GROUP_COL)
    total_group = len(grouped_future)

    print(f"\\nMemproses {total_group} groups secara paralel "
          f"({forecast_label})...")

    start_time = time.time()

    # --- Parallel execution ---
    # Setiap group diproses independen oleh worker berbeda
    results = Parallel(n_jobs=n_jobs, verbose=1, backend="loky")(
        delayed(process_group)(
            kab_kota, kab_future_df, history_df, horizon
        )
        for kab_kota, kab_future_df in grouped_future
    )

    # --- Gabungkan semua hasil ---
    all_predictions = []
    for group_preds in results:
        all_predictions.extend(group_preds)

    elapsed = time.time() - start_time
    print(f"{forecast_label} selesai dalam {elapsed:.1f} detik "
          f"({elapsed/60:.1f} menit). "
          f"Total prediksi: {len(all_predictions)}")

    pred_df = pd.DataFrame(all_predictions)
    return pred_df


# =========================================================
# VALIDATION RECURSIVE FORECAST
# =========================================================

print("\\n" + "=" * 80)
print("VALIDATION RECURSIVE FORECAST")
print("=" * 80)

val_predictions = recursive_forecast(
    model=model,
    history_df=train_df,
    future_df=val_df,
    horizon=FORECAST_HORIZON,
    forecast_label="VALIDATION",
)

# =========================================================
# TEST RECURSIVE FORECAST
# =========================================================

print("\\n" + "=" * 80)
print("TEST RECURSIVE FORECAST")
print("=" * 80)

full_history_for_test = pd.concat(
    [train_df, val_df],
    ignore_index=True,
)

test_predictions = recursive_forecast(
    model=model,
    history_df=full_history_for_test,
    future_df=test_df,
    horizon=FORECAST_HORIZON,
    forecast_label="TEST",
)

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def calculate_metrics(pred_df):

    results = []

    for h in range(1, FORECAST_HORIZON + 1):

        temp = pred_df[
            pred_df["horizon"] == h
        ].copy()

        y_true = temp["actual"]
        y_pred = temp["prediction"]

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mape = np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        ) * 100

        r2 = r2_score(
            y_true,
            y_pred
        )

        results.append({
            "Horizon": f"H+{h}",
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
            "R2": round(r2, 4),
        })

    return pd.DataFrame(results)


# =========================================================
# VALIDATION METRICS
# =========================================================

print("\\n" + "=" * 80)
print("VALIDATION METRICS")
print("=" * 80)

val_metrics = calculate_metrics(
    val_predictions
)

print(val_metrics)

# =========================================================
# TEST METRICS
# =========================================================

print("\\n" + "=" * 80)
print("TEST METRICS")
print("=" * 80)

test_metrics = calculate_metrics(
    test_predictions
)

print(test_metrics)

# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

print("\\n" + "=" * 80)
print("SAVE TEST PREDICTIONS")
print("=" * 80)

test_predictions.to_csv(
    "training/cabai rawit/test_predictions_cabai_rawit.csv",
    index=False,
)
print(
    "Test predictions saved -> "
    "training/cabai rawit/test_predictions_cabai_rawit.csv"
    f"  ({len(test_predictions):,} rows)"
)

val_predictions.to_csv(
    "training/cabai rawit/val_predictions_cabai_rawit.csv",
    index=False,
)
print(
    "Validation predictions saved -> "
    "training/cabai rawit/val_predictions_cabai_rawit.csv"
    f"  ({len(val_predictions):,} rows)"
)

test_metrics.to_csv(
    "training/cabai rawit/test_metrics_cabai_rawit.csv",
    index=False,
)
print(
    "Test metrics saved -> "
    "training/cabai rawit/test_metrics_cabai_rawit.csv"
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("\\n" + "=" * 80)
print("TOP FEATURE IMPORTANCE")
print("=" * 80)

importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

print(
    importance_df.head(30)
)
"""