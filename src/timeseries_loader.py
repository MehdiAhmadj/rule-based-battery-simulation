# timeseries_loader.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Literal

# ---------- Public API -------------------------------------------------

def load_hourly_price_and_windspeed(
    price_csv: str = "Dataset/electricityprice.csv",
    wind_csv: str = "Dataset/windspeed.csv",
    mode: Literal["range", "season"] = "range",
    *,
    # Range mode
    start: Optional[str] = None,     # e.g., "2021-01-01 00:00"
    end: Optional[str] = None,       # e.g., "2021-01-07 23:59"
    # Season mode
    season: Optional[Literal["winter", "spring", "summer", "autumn"]] = None,
    year: Optional[int] = None,
    n_weeks: Optional[int] = None,
    week_offset: int = 0,
    #
    timezone: Optional[str] = None,  # e.g., "Europe/Berlin"

    # ---- Flexible column specs (accept either explicit or candidates) ----
    # PRICE time/value columns
    price_time_col: Optional[str] = None,
    price_time_col_candidates: Tuple[str, ...] = ("datetime", "dataset", "time", "timestamp"),
    price_col_candidates: Tuple[str, ...] = ("price", "price[€/MWh]", "price_eur_mwh"),
    # WIND time/value columns
    wind_time_col: Optional[str] = None,
    wind_time_col_candidates: Tuple[str, ...] = ("datetime", "dataset", "time", "timestamp"),
    wind_col_candidates: Tuple[str, ...] = ("wind_speed", "windspeed", "v", "speed"),

    # Resampling
    agg: Literal["mean", "median"] = "mean",
) -> Dict[str, object]:
    """
    Returns dict with:
      - prices_series, wind_series (hourly)
      - PRICES_EUR_PER_MWH, WIND_SPEEDS (lists of float)
      - T_HOURS (int), index (DatetimeIndex), meta (dict)
    """

    # ---- Load raw 10-min series using robust internal loader ----
    prices_10m = _load_timeseries(
        price_csv,
        time_col=price_time_col,
        time_col_candidates=price_time_col_candidates,
        value_col_candidates=price_col_candidates,
        timezone=timezone,
    )
    wind_10m = _load_timeseries(
        wind_csv,
        time_col=wind_time_col,
        time_col_candidates=wind_time_col_candidates,
        value_col_candidates=wind_col_candidates,
        timezone=timezone,
    )

    # Resample to hourly
    prices_1h = _resample_hourly(prices_10m, agg=agg)
    wind_1h   = _resample_hourly(wind_10m,   agg=agg)

    # Select window
    if mode == "range":
        if start is None or end is None:
            raise ValueError("When mode='range', provide both start and end.")
        prices_sel = prices_1h.loc[start:end]
        wind_sel   = wind_1h.loc[start:end]
    elif mode == "season":
        if season is None or year is None or n_weeks is None:
            raise ValueError("When mode='season', provide season, year, and n_weeks.")
        season_start, season_end = _season_bounds(year, season)
        start_dt = season_start + pd.to_timedelta(7 * week_offset, unit="D")
        end_dt   = start_dt + pd.to_timedelta(7 * n_weeks, unit="D") - pd.to_timedelta(1, unit="H")
        prices_sel = prices_1h.loc[start_dt:end_dt]
        wind_sel   = wind_1h.loc[start_dt:end_dt]
    else:
        raise ValueError("mode must be 'range' or 'season'.")

    # Align and drop NaNs
    df = pd.concat({"price": prices_sel, "wind": wind_sel}, axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping hourly data after selection.")

    prices_series = df["price"].astype(float)
    wind_series   = df["wind"].astype(float)

    return {
        "prices_series": prices_series,
        "wind_series": wind_series,
        "PRICES_EUR_PER_MWH": prices_series.tolist(),
        "WIND_SPEEDS": wind_series.tolist(),
        "T_HOURS": len(prices_series),
        "index": prices_series.index,
        "meta": {
            "mode": mode,
            "start": str(prices_series.index.min()),
            "end": str(prices_series.index.max()),
            "agg": agg,
            "season": season,
            "year": year,
            "n_weeks": n_weeks,
            "week_offset": week_offset,
            "timezone": timezone,
            "rows_hourly": len(df),
        },
    }

# ---------- Internals ---------------------------------------------------

def _read_csv_robust(path: str):
    """
    Try multiple encodings and flexible separator inference.
    Returns a pandas DataFrame.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin1")
    last_err = None
    for enc in encodings:
        try:
            # Use engine="python" + sep=None to sniff commas/semicolons/tabs
            df = pd.read_csv(
                path,
                engine="python",
                sep=None,              # auto-detect delimiter
                # infer_datetime_format is deprecated; default is strict now
                on_bad_lines="skip",   # tolerate a few bad rows
            )
            # Ensure the raw text was decoded with this encoding; if it fails, an exception is thrown earlier
            df.columns = [str(c) for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f"Failed to read CSV: {path}")

def _load_timeseries(
    path: str,
    *,
    # Backward compatible: you may pass EITHER a single time_col OR time_col_candidates
    time_col: str | None = None,
    time_col_candidates: Tuple[str, ...] | None = None,
    value_col_candidates: Tuple[str, ...],
    timezone: Optional[str] = None,
) -> pd.Series:
    """
    Robust CSV loader:
      - tries multiple encodings (utf-8, utf-8-sig, cp1252, latin1),
      - auto-detects delimiter (comma/semicolon/tab) with engine='python',
      - accepts either a specific time_col or a list of candidates,
      - parses datetimes and coerces numeric values to float,
      - optional timezone localization/conversion.
    """
    # 1) Read CSV robustly (encoding + delimiter sniffing)
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin1")
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                engine="python",   # robust parser
                sep=None,          # auto-detect delimiter
                encoding=enc,
                on_bad_lines="skip",
            )
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err if last_err else RuntimeError(f"Failed to read CSV: {path}")

    # 2) Decide time column
    chosen_time_col = None
    if time_col is not None:
        if time_col in df.columns:
            chosen_time_col = time_col
        else:
            raise ValueError(
                f"time_col='{time_col}' not found in {path}. Columns: {list(df.columns)}"
            )
    else:
        # try candidates (if provided)
        if time_col_candidates:
            for c in time_col_candidates:
                if c in df.columns:
                    chosen_time_col = c
                    break
        # fallback to the first column
        if chosen_time_col is None:
            chosen_time_col = df.columns[0]

    # 3) Decide value column
    val_col = None
    for c in value_col_candidates:
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        # fallback to the last column if none matched
        val_col = df.columns[-1]

    # 4) Parse datetime + clean
    s = df[[chosen_time_col, val_col]].copy()
    s[chosen_time_col] = pd.to_datetime(s[chosen_time_col], errors="coerce")  # no infer_datetime_format
    s = s.dropna(subset=[chosen_time_col])
    s = s.set_index(chosen_time_col).sort_index()

    # 5) Numeric coercion
    s[val_col] = pd.to_numeric(s[val_col], errors="coerce")
    s = s.dropna(subset=[val_col])

    # 6) Timezone handling (optional)
    if timezone is not None:
        if s.index.tz is None:
            s.index = s.index.tz_localize(timezone, nonexistent="NaT", ambiguous="NaT")
        else:
            s.index = s.index.tz_convert(timezone)
        s = s[~s.index.isna()]

    return s[val_col].astype(float)




def _resample_hourly(s: pd.Series, agg: str = "mean") -> pd.Series:
    """Resample sub-hourly series to hourly using mean/median."""
    if agg == "mean":
        return s.resample("1H").mean()
    elif agg == "median":
        return s.resample("1H").median()
    else:
        raise ValueError("agg must be 'mean' or 'median'")

def _season_bounds(year: int, season: Literal["winter", "spring", "summer", "autumn"]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Meteorological seasons (UTC-like boundaries)."""
    if season == "winter":
        start = pd.Timestamp(year=year-1, month=12, day=1, hour=0)
        end   = pd.Timestamp(year=year,   month=2,  day=28, hour=23)
    elif season == "spring":
        start = pd.Timestamp(year=year, month=3, day=1, hour=0)
        end   = pd.Timestamp(year=year, month=5, day=31, hour=23)
    elif season == "summer":
        start = pd.Timestamp(year=year, month=6, day=1, hour=0)
        end   = pd.Timestamp(year=year, month=8, day=31, hour=23)
    elif season == "autumn":
        start = pd.Timestamp(year=year, month=9, day=1, hour=0)
        end   = pd.Timestamp(year=year, month=11, day=30, hour=23)
    else:
        raise ValueError("season must be one of: winter, spring, summer, autumn")
    return start, end

# ---------- Quick examples (optional) -----------------------------------

def example_range():
    return load_hourly_price_and_windspeed(
        price_csv="Dataset/electricityprice.csv",
        wind_csv="Dataset/windspeed.csv",
        mode="range",
        start="2021-01-01 00:00",
        end="2021-01-07 23:59",
    )

def example_season():
    return load_hourly_price_and_windspeed(
        price_csv="Dataset/electricityprice.csv",
        wind_csv="Dataset/windspeed.csv",
        mode="season",
        season="summer",
        year=2022,
        n_weeks=2,
        week_offset=1,
    )
