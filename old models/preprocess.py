"""Leakage-safe preprocessing for the Beijing multi-site air-quality competition.

This version matches the ACTUAL repository CSV schema, which does not contain
`current_PM2_5` even though the README currently lists that field.

It is designed for CatBoost / LightGBM / XGBoost and creates only features that
can be reproduced for both train.csv and test.csv without using hidden targets.

Usage from repository root:
    python preprocess.py

Optional:
    python preprocess.py --train train.csv --test test.csv --out-dir data/processed
    python preprocess.py --csv

Dependencies:
    pip install pandas numpy pyarrow
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

TARGET = "PM2_5_next_hour"
ID_COL = "id"
TIME_COL = "observation_timestamp"
STATION_COL = "station"
WIND_COL = "wd"

# Exact columns present in the competition files.
POLLUTANTS = ["PM10", "SO2", "NO2", "CO", "O3"]
METEO = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
BASE_NUMERIC = POLLUTANTS + METEO

# Richer history for PM10 because it is the closest available particulate
# measurement to PM2.5. Other variables get a slightly smaller history.
PM10_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
OTHER_LAGS = [1, 2, 3, 6, 12, 24, 48]

PM10_ROLL_WINDOWS = [3, 6, 12, 24, 48, 72, 168]
POLLUTANT_ROLL_WINDOWS = [3, 6, 12, 24]
METEO_ROLL_WINDOWS = [6, 12, 24]

WIND_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=Path("train.csv"))
    p.add_argument("--test", type=Path, default=Path("test.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--csv",
        action="store_true",
        help="Save CSV instead of Parquet. Parquet is recommended.",
    )
    return p.parse_args()


def validate_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    # Trim accidental spaces in headers first.
    train.columns = [str(c).strip() for c in train.columns]
    test.columns = [str(c).strip() for c in test.columns]

    required_common = {
        ID_COL, TIME_COL, STATION_COL, WIND_COL,
        "year", "month", "day", "hour", *BASE_NUMERIC,
    }
    required_train = required_common | {TARGET}

    missing_train = sorted(required_train - set(train.columns))
    missing_test = sorted(required_common - set(test.columns))
    if missing_train or missing_test:
        raise ValueError(
            f"Missing columns. train={missing_train or 'none'}, "
            f"test={missing_test or 'none'}"
        )

    if train[ID_COL].duplicated().any():
        raise ValueError("train.csv contains duplicate IDs")
    if test[ID_COL].duplicated().any():
        raise ValueError("test.csv contains duplicate IDs")


def prepare_raw(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    test = test.copy()

    train[TIME_COL] = pd.to_datetime(train[TIME_COL], errors="raise")
    test[TIME_COL] = pd.to_datetime(test[TIME_COL], errors="raise")

    train["__source"] = "train"
    test["__source"] = "test"
    train["__row"] = np.arange(len(train), dtype=np.int64)
    test["__row"] = np.arange(len(test), dtype=np.int64)

    # Placeholder only for concatenation. Hidden target is never used.
    if TARGET not in test.columns:
        test[TARGET] = np.nan

    df = pd.concat([train, test], ignore_index=True, sort=False)

    df[STATION_COL] = df[STATION_COL].fillna("__MISSING__").astype(str)
    df[WIND_COL] = df[WIND_COL].fillna("__MISSING__").astype(str)

    for col in BASE_NUMERIC + ["year", "month", "day", "hour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # All historical feature engineering is chronological within station.
    df = df.sort_values([STATION_COL, TIME_COL, "__source", "__row"]).reset_index(drop=True)

    duplicate_station_time = df.duplicated([STATION_COL, TIME_COL], keep=False)
    if duplicate_station_time.any():
        n = int(duplicate_station_time.sum())
        raise ValueError(
            f"Found {n} rows involved in duplicate station/timestamp pairs. "
            "Inspect the raw data before creating lag features."
        )

    return df


def add_calendar_features(df: pd.DataFrame) -> None:
    ts = df[TIME_COL]

    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["day_of_year"] = ts.dt.dayofyear.astype("int16")
    df["week_of_year"] = ts.dt.isocalendar().week.astype("int16")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * (df["day_of_year"] - 1) / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * (df["day_of_year"] - 1) / 365.25)

    global_start = ts.min()
    df["hours_from_start"] = (ts - global_start).dt.total_seconds() / 3600.0

    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype("int8")
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
    df["season"] = ((df["month"] % 12) // 3).astype("int8")


def add_wind_features(df: pd.DataFrame) -> None:
    deg = df[WIND_COL].map(WIND_TO_DEG)
    rad = np.deg2rad(deg)

    df["wd_deg"] = deg
    df["wd_sin"] = np.sin(rad)
    df["wd_cos"] = np.cos(rad)
    df["wind_x"] = df["WSPM"] * df["wd_cos"]
    df["wind_y"] = df["WSPM"] * df["wd_sin"]
    df["is_calm"] = (df["WSPM"] < 1.0).astype("int8")
    df["wind_stagnation"] = 1.0 / (1.0 + df["WSPM"].clip(lower=0))


def add_missingness_and_causal_fill(df: pd.DataFrame) -> None:
    for col in BASE_NUMERIC:
        df[f"{col}_missing"] = df[col].isna().astype("int8")

    df["missing_count"] = df[BASE_NUMERIC].isna().sum(axis=1).astype("int8")
    df["any_missing"] = (df["missing_count"] > 0).astype("int8")

    # Preserve raw NaNs. These extra columns only expose the most recent known
    # measurement when it is no more than 6 clock-hours old.
    grouped = df.groupby(STATION_COL, sort=False)
    for col in BASE_NUMERIC:
        filled = grouped[col].ffill()
        observed_time = df[TIME_COL].where(df[col].notna())
        last_time = observed_time.groupby(df[STATION_COL], sort=False).ffill()
        age_hours = (df[TIME_COL] - last_time).dt.total_seconds() / 3600.0
        df[f"{col}_ffill6"] = filled.where(age_hours <= 6.0)
        df[f"{col}_age_hours"] = age_hours


def add_physics_and_interactions(df: pd.DataFrame) -> None:
    eps = 1e-3

    df["temp_dewp_spread"] = df["TEMP"] - df["DEWP"]

    # Approximate RH (%) from temperature and dew point.
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(
        (a * df["DEWP"] / (b + df["DEWP"]))
        - (a * df["TEMP"] / (b + df["TEMP"]))
    )
    df["relative_humidity_approx"] = rh.clip(0, 100)

    df["is_raining"] = (df["RAIN"] > 0).astype("int8")
    df["rain_log1p"] = np.log1p(df["RAIN"].clip(lower=0))

    # Cross-pollutant chemistry / source proxies.
    df["pm10_log1p"] = np.log1p(df["PM10"].clip(lower=0))
    df["so2_log1p"] = np.log1p(df["SO2"].clip(lower=0))
    df["no2_log1p"] = np.log1p(df["NO2"].clip(lower=0))
    df["co_log1p"] = np.log1p(df["CO"].clip(lower=0))
    df["o3_log1p"] = np.log1p(df["O3"].clip(lower=0))

    df["pm10_to_co"] = df["PM10"] / (df["CO"].abs() + eps)
    df["pm10_to_no2"] = df["PM10"] / (df["NO2"].abs() + eps)
    df["no2_to_o3"] = df["NO2"] / (df["O3"].abs() + eps)
    df["so2_to_no2"] = df["SO2"] / (df["NO2"].abs() + eps)
    df["oxidant_sum"] = df["NO2"] + df["O3"]
    df["combustion_index"] = df["NO2"] + df["SO2"] + df["CO"]

    # Stagnant air often supports particle accumulation.
    df["pm10_wind_interaction"] = df["PM10"] / (1.0 + df["WSPM"].clip(lower=0))
    df["no2_wind_interaction"] = df["NO2"] / (1.0 + df["WSPM"].clip(lower=0))
    df["humidity_pm10_interaction"] = df["PM10"] * df["relative_humidity_approx"] / 100.0


def add_network_features(df: pd.DataFrame) -> None:
    """Same-hour cross-station context using only available predictors."""
    by_time = df.groupby(TIME_COL, sort=False)

    for col in POLLUTANTS + ["TEMP", "PRES", "DEWP", "WSPM"]:
        x = df[col]
        grp = by_time[col]
        total = grp.transform("sum")
        count = grp.transform("count")

        own_present = x.notna().astype("int16")
        loo_count = count - own_present
        loo_sum = total - x.fillna(0.0)
        loo_mean = loo_sum / loo_count.replace(0, np.nan)

        df[f"network_{col}_mean"] = grp.transform("mean")
        df[f"network_{col}_median"] = grp.transform("median")
        df[f"network_{col}_std"] = grp.transform("std")
        df[f"network_{col}_loo_mean"] = loo_mean
        df[f"{col}_vs_network"] = x - loo_mean


def exact_lag_values(df: pd.DataFrame, col: str, lag_hours: int) -> np.ndarray:
    """Value from the same station at exactly t-lag_hours."""
    current_index = pd.MultiIndex.from_frame(df[[STATION_COL, TIME_COL]])
    source = df.set_index([STATION_COL, TIME_COL])[col].copy()

    shifted_index = pd.MultiIndex.from_arrays(
        [
            source.index.get_level_values(0),
            source.index.get_level_values(1) + pd.Timedelta(hours=lag_hours),
        ],
        names=[STATION_COL, TIME_COL],
    )
    source.index = shifted_index
    return source.reindex(current_index).to_numpy()


def add_lag_features(df: pd.DataFrame) -> None:
    grouped = df.groupby(STATION_COL, sort=False)
    prev_ts = grouped[TIME_COL].shift(1)
    df["hours_since_prev_observation"] = (
        (df[TIME_COL] - prev_ts).dt.total_seconds() / 3600.0
    )
    df["has_time_gap"] = (df["hours_since_prev_observation"] > 1.0).astype("int8")

    # PM10: richest temporal history.
    for lag in PM10_LAGS:
        df[f"PM10_lag{lag}"] = exact_lag_values(df, "PM10", lag)

    # Other measured pollutants + weather.
    for col in ["SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
        for lag in OTHER_LAGS:
            df[f"{col}_lag{lag}"] = exact_lag_values(df, col, lag)

    # Trends / slopes use only current and previous measurements.
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"PM10_delta_{lag}h"] = df["PM10"] - df[f"PM10_lag{lag}"]
        df[f"PM10_slope_{lag}h"] = df[f"PM10_delta_{lag}h"] / float(lag)
        df[f"PM10_linear_forecast_{lag}h"] = df["PM10"] + df[f"PM10_slope_{lag}h"]

    df["PM10_acceleration_1h"] = (
        df["PM10"] - 2.0 * df["PM10_lag1"] + df["PM10_lag2"]
    )

    for col in ["NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]:
        df[f"{col}_delta1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_delta3"] = df[col] - df[f"{col}_lag3"]
        df[f"{col}_delta24"] = df[col] - df[f"{col}_lag24"]


def rolling_transform(
    df: pd.DataFrame,
    col: str,
    window: int,
    stats: tuple[str, ...],
) -> None:
    # Row windows are safe because the rows are sorted chronologically by station.
    # Exact lag features above separately preserve exact clock-hour history.
    g = df.groupby(STATION_COL, sort=False)[col]
    rolled = g.rolling(window=window, min_periods=1)

    if "mean" in stats:
        df[f"{col}_roll{window}_mean"] = (
            rolled.mean().reset_index(level=0, drop=True).sort_index()
        )
    if "std" in stats:
        df[f"{col}_roll{window}_std"] = (
            rolled.std().reset_index(level=0, drop=True).sort_index()
        )
    if "min" in stats:
        df[f"{col}_roll{window}_min"] = (
            rolled.min().reset_index(level=0, drop=True).sort_index()
        )
    if "max" in stats:
        df[f"{col}_roll{window}_max"] = (
            rolled.max().reset_index(level=0, drop=True).sort_index()
        )


def add_rolling_features(df: pd.DataFrame) -> None:
    for w in PM10_ROLL_WINDOWS:
        rolling_transform(df, "PM10", w, ("mean", "std", "min", "max"))

    for col in ["SO2", "NO2", "CO", "O3"]:
        for w in POLLUTANT_ROLL_WINDOWS:
            rolling_transform(df, col, w, ("mean", "std"))

    for col in ["TEMP", "PRES", "DEWP", "WSPM"]:
        for w in METEO_ROLL_WINDOWS:
            rolling_transform(df, col, w, ("mean", "std"))

    g = df.groupby(STATION_COL, sort=False)
    for col in ["PM10", "NO2", "CO", "O3"]:
        for span in [3, 6, 12, 24]:
            df[f"{col}_ewm{span}"] = g[col].transform(
                lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean()
            )

    for w in PM10_ROLL_WINDOWS:
        df[f"PM10_vs_roll{w}"] = df["PM10"] - df[f"PM10_roll{w}_mean"]


def downcast(df: pd.DataFrame) -> None:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in ["year", "month", "day", "hour"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")


def save_outputs(
    df: pd.DataFrame,
    out_dir: Path,
    save_csv: bool,
    raw_train_missing: dict[str, float],
    raw_test_missing: dict[str, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = (
        df[df["__source"] == "train"]
        .sort_values("__row")
        .drop(columns=["__source", "__row"])
        .reset_index(drop=True)
    )
    test_out = (
        df[df["__source"] == "test"]
        .sort_values("__row")
        .drop(columns=["__source", "__row", TARGET], errors="ignore")
        .reset_index(drop=True)
    )

    categorical = [STATION_COL, WIND_COL]
    excluded = {ID_COL, TIME_COL, TARGET}
    feature_cols = [c for c in train_out.columns if c not in excluded]

    metadata = {
        "target": TARGET,
        "id_column": ID_COL,
        "timestamp_column": TIME_COL,
        "categorical_features": categorical,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "train_rows": len(train_out),
        "test_rows": len(test_out),
        "train_time_min": str(train_out[TIME_COL].min()),
        "train_time_max": str(train_out[TIME_COL].max()),
        "test_time_min": str(test_out[TIME_COL].min()),
        "test_time_max": str(test_out[TIME_COL].max()),
        "raw_train_missing_fraction": raw_train_missing,
        "raw_test_missing_fraction": raw_test_missing,
        "notes": [
            "Matches actual CSV schema: current_PM2_5 is not present.",
            "No feature is derived from PM2_5_next_hour.",
            "Use chronological validation, not random train_test_split.",
            "Numeric NaNs are preserved; missing flags and short causal ffill features are added.",
            "Train and test predictors are concatenated only for causal predictor-history features.",
        ],
    }

    with (out_dir / "feature_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if save_csv:
        train_path = out_dir / "train_processed.csv"
        test_path = out_dir / "test_processed.csv"
        train_out.to_csv(train_path, index=False)
        test_out.to_csv(test_path, index=False)
    else:
        train_path = out_dir / "train_processed.parquet"
        test_path = out_dir / "test_processed.parquet"
        try:
            train_out.to_parquet(train_path, index=False)
            test_out.to_parquet(test_path, index=False)
        except ImportError as e:
            raise SystemExit(
                "Parquet support is missing. Run 'pip install pyarrow' or rerun with --csv."
            ) from e

    print("\nPreprocessing complete")
    print(f"  train: {train_path}  shape={train_out.shape}")
    print(f"  test : {test_path}  shape={test_out.shape}")
    print(f"  model features: {len(feature_cols)}")
    print(f"  metadata: {out_dir / 'feature_metadata.json'}")
    print("\nImportant: validate chronologically; do NOT randomly shuffle the rows.")


def main() -> None:
    args = parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    print("Train columns:", train.columns.tolist())
    print("Test columns :", test.columns.tolist())

    validate_columns(train, test)

    raw_train_missing = train[BASE_NUMERIC + [WIND_COL]].isna().mean().round(6).to_dict()
    raw_test_missing = test[BASE_NUMERIC + [WIND_COL]].isna().mean().round(6).to_dict()

    print(f"Raw train shape: {train.shape}")
    print(f"Raw test shape : {test.shape}")
    print("\nTop train missing fractions:")
    print(train.isna().mean().sort_values(ascending=False).head(12))

    df = prepare_raw(train, test)

    add_calendar_features(df)
    add_wind_features(df)
    add_missingness_and_causal_fill(df)
    add_physics_and_interactions(df)
    add_network_features(df)
    add_lag_features(df)
    add_rolling_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    downcast(df)

    save_outputs(
        df=df,
        out_dir=args.out_dir,
        save_csv=args.csv,
        raw_train_missing=raw_train_missing,
        raw_test_missing=raw_test_missing,
    )


if __name__ == "__main__":
    main()
