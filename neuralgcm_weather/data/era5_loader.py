"""
ERA5 Historical Data Loader
============================
Loads ERA5 reanalysis data from Google Cloud Storage.
Used for:
  - Historical forecasts (1979-2020)
  - Verification / ground truth comparison
  - Seasonal analogs when real-time data unavailable
"""

import numpy as np
import xarray as xr
import pandas as pd
from loguru import logger
from typing import Optional, List

ERA5_ZARR = ("gs://gcp-public-data-arco-era5/ar/"
             "full_37-1h-0p25deg-chunk-1.zarr-v3")
ERA5_MIN_DATE = pd.Timestamp("1979-01-01")
ERA5_MAX_DATE = pd.Timestamp("2020-12-31")


def validate_date(dt: pd.Timestamp) -> pd.Timestamp:
    """Clamp date to ERA5 available range with informative warning."""
    if dt < ERA5_MIN_DATE:
        logger.warning(f"Date {dt} before ERA5 start. Using {ERA5_MIN_DATE}")
        return ERA5_MIN_DATE
    if dt > ERA5_MAX_DATE:
        logger.warning(
            f"Date {dt} after ERA5 cutoff ({ERA5_MAX_DATE}). "
            f"For real-time forecasts use ecmwf_loader.py instead.")
        return ERA5_MAX_DATE
    return dt


def open_era5(zarr_url: str = ERA5_ZARR) -> xr.Dataset:
    """Open ERA5 zarr dataset (lazy, no download)."""
    import gcsfs
    logger.info("Opening ERA5 zarr dataset (lazy)...")
    era5 = xr.open_zarr(
        zarr_url,
        chunks=None,
        storage_options=dict(token="anon"),
    )
    logger.success(f"ERA5 opened | dims={dict(era5.sizes)}")
    return era5


def load_era5_slice(
    era5: xr.Dataset,
    init_date: str,
    variables: List[str],
) -> xr.Dataset:
    """
    Load a single ERA5 time slice for model initialisation.
    Computes into memory — safe for single init times.
    """
    dt = validate_date(pd.Timestamp(init_date))
    logger.info(f"Loading ERA5 slice at {dt}...")

    available = [v for v in variables if v in era5.data_vars]
    missing   = [v for v in variables if v not in era5.data_vars]
    if missing:
        logger.warning(f"Variables not in ERA5: {missing}")

    slc = (era5[available]
           .sel(time=dt, method="nearest")
           .compute())
    logger.success(
        f"ERA5 slice loaded | "
        f"dims={dict(slc.sizes)} | "
        f"vars={list(slc.data_vars)}")
    return slc


def get_seasonal_analog(target_date: str,
                        year: int = 2020) -> str:
    """
    Return the equivalent historical date from a given year.
    Used when real-time ECMWF data is unavailable.
    E.g. "2026-03-25" -> "2020-03-25" (same day, same season)
    """
    dt = pd.Timestamp(target_date)
    analog = dt.replace(year=year)
    analog = validate_date(analog)
    logger.info(
        f"Seasonal analog: {target_date} -> "
        f"{analog.strftime('%Y-%m-%d')} (year {year})")
    return analog.strftime("%Y-%m-%dT00:00")


def load_era5_point_series(
    era5: xr.Dataset,
    var: str,
    lat: float,
    lon: float,
    dates: List[pd.Timestamp],
    level: Optional[int] = None,
) -> np.ndarray:
    """
    Extract ERA5 truth values at a point for a list of dates.
    Used for forecast verification (computing MAE vs ERA5).
    """
    lon_360 = lon % 360
    lat_coord = "latitude" if "latitude" in era5.coords else "lat"
    lon_coord = "longitude" if "longitude" in era5.coords else "lon"

    vals = []
    for dt in dates:
        try:
            da = era5[var].sel(time=dt, method="nearest")
            if level is not None and "level" in da.dims:
                da = da.sel(level=level, method="nearest")
            da = da.sel(
                {lat_coord: lat, lon_coord: lon_360},
                method="nearest")
            vals.append(float(da.compute()))
        except Exception as e:
            logger.warning(f"ERA5 point extraction failed at {dt}: {e}")
            vals.append(np.nan)
    return np.array(vals)
