"""
ECMWF Open Data Loader
======================
Downloads today's ECMWF operational analysis (public, no auth needed).
ECMWF releases data with a ~6 hour lag.
This enables NeuralGCM forecasts initialised from TODAY's atmosphere,
not historical ERA5.

Data source: https://data.ecmwf.int/forecasts/
License: CC-4.0 (free for commercial use)
"""

import os
import time
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional

# NeuralGCM 2.8deg requires these pressure-level variables
# mapped to ERA5 naming convention
ECMWF_VARS = [
    "u",      # u_component_of_wind
    "v",      # v_component_of_wind
    "t",      # temperature
    "q",      # specific_humidity
    "z",      # geopotential
    "clwc",   # specific_cloud_liquid_water_content
    "ciwc",   # specific_cloud_ice_water_content
    "w",      # vertical_velocity (optional)
]

ECMWF_TO_ERA5 = {
    "u":    "u_component_of_wind",
    "v":    "v_component_of_wind",
    "t":    "temperature",
    "q":    "specific_humidity",
    "z":    "geopotential",
    "clwc": "specific_cloud_liquid_water_content",
    "ciwc": "specific_cloud_ice_water_content",
    "w":    "vertical_velocity",
    "sp":   "surface_pressure",
    "lnsp": "log_surface_pressure",
}

PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 150, 200, 250, 300, 400, 500,
    600, 700, 850, 925, 1000
]


def get_latest_available_date(lag_hours: int = 6) -> pd.Timestamp:
    """
    ECMWF open data has a ~6h publication lag.
    Returns the most recent date/time we can safely request.
    """
    now = pd.Timestamp.utcnow()
    # Round down to nearest 6h cycle (00, 06, 12, 18 UTC)
    cycle_hour = (now.hour - lag_hours) // 6 * 6
    if cycle_hour < 0:
        now -= pd.Timedelta(days=1)
        cycle_hour = 18
    return now.replace(
        hour=cycle_hour, minute=0, second=0, microsecond=0)


def download_ecmwf_analysis(
    target_date: Optional[str] = None,
    cache_dir: str = "./cache",
    lag_hours: int = 6,
) -> Optional[Path]:
    """
    Download ECMWF operational analysis for a given date.
    Falls back to next-most-recent date if requested date unavailable.

    Args:
        target_date: "YYYY-MM-DD" or None for latest available
        cache_dir:   local directory for caching GRIB files
        lag_hours:   ECMWF publication lag in hours

    Returns:
        Path to downloaded GRIB file, or None if download failed
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Determine target date
    if target_date is None:
        dt = get_latest_available_date(lag_hours)
    else:
        dt = pd.Timestamp(target_date)

    date_str = dt.strftime("%Y%m%d")
    time_str = f"{dt.hour:02d}0000"
    grib_file = Path(cache_dir) / f"ecmwf_pl_{date_str}_{time_str}.grib2"

    if grib_file.exists():
        logger.info(f"Using cached ECMWF file: {grib_file}")
        return grib_file

    logger.info(
        f"Downloading ECMWF analysis: {dt.strftime('%Y-%m-%d %H:%M UTC')}")

    # Try ecmwf-opendata client first (preferred)
    try:
        from ecmwf.opendata import Client
        client = Client("ecmwf")

        # Download pressure-level fields
        client.retrieve(
            date=date_str,
            time=f"{dt.hour:02d}",
            step=0,
            stream="oper",
            type="an",
            param=ECMWF_VARS,
            levtype="pl",
            levelist="/".join(str(l) for l in PRESSURE_LEVELS),
            target=str(grib_file),
        )
        logger.success(f"Downloaded: {grib_file} "
                       f"({grib_file.stat().st_size / 1e6:.1f} MB)")
        return grib_file

    except Exception as e:
        logger.warning(f"ecmwf-opendata failed: {e}")

    # Fallback: try direct HTTP download from ECMWF open data
    try:
        import urllib.request
        base = "https://data.ecmwf.int/forecasts"
        url  = (f"{base}/{date_str}/{time_str}z/0p4-beta/"
                f"oper/0/{date_str}{time_str}0000-0h-oper-fc.grib2")
        logger.info(f"Trying direct download: {url}")
        urllib.request.urlretrieve(url, str(grib_file))
        logger.success(f"Downloaded via HTTP: {grib_file}")
        return grib_file

    except Exception as e:
        logger.error(f"Direct download failed: {e}")

    # Final fallback: try previous 6h cycle
    if target_date is None:
        logger.warning("Trying previous 6h cycle...")
        prev_dt = dt - pd.Timedelta(hours=6)
        return download_ecmwf_analysis(
            target_date=prev_dt.strftime("%Y-%m-%d"),
            cache_dir=cache_dir,
            lag_hours=0,
        )

    return None


def grib_to_xarray(grib_file: Path) -> Optional[xr.Dataset]:
    """
    Convert ECMWF GRIB2 file to xarray Dataset matching ERA5 format.
    Handles variable renaming and coordinate standardisation.
    """
    logger.info(f"Reading GRIB: {grib_file}")
    try:
        import cfgrib
    except ImportError:
        logger.error("cfgrib not installed. Run: pip install cfgrib eccodes")
        return None

    datasets = []

    # Open pressure-level data
    try:
        ds_pl = xr.open_dataset(
            str(grib_file),
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
                "errors": "ignore",
            },
            indexpath=str(grib_file) + ".idx",
        )
        datasets.append(ds_pl)
        logger.info(f"  Pressure-level vars: {list(ds_pl.data_vars)}")
    except Exception as e:
        logger.warning(f"  Pressure-level read failed: {e}")

    # Open surface data (surface pressure, 2m temp etc.)
    try:
        ds_sfc = xr.open_dataset(
            str(grib_file),
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "surface"},
                "errors": "ignore",
            },
        )
        datasets.append(ds_sfc)
        logger.info(f"  Surface vars: {list(ds_sfc.data_vars)}")
    except Exception as e:
        logger.warning(f"  Surface read failed: {e}")

    if not datasets:
        logger.error("Could not read any data from GRIB file")
        return None

    # Merge all datasets
    try:
        ds = xr.merge(datasets, compat="override")
    except Exception:
        ds = datasets[0]

    # Rename variables to ERA5 convention
    rename = {k: v for k, v in ECMWF_TO_ERA5.items()
              if k in ds.data_vars}
    ds = ds.rename(rename)
    logger.info(f"  Renamed to ERA5 format: {list(ds.data_vars)}")

    # Standardise coordinate names
    coord_rename = {}
    for c in ds.coords:
        cl = c.lower()
        if cl in ("lat", "latitude_0"):
            coord_rename[c] = "latitude"
        elif cl in ("lon", "longitude_0"):
            coord_rename[c] = "longitude"
        elif cl in ("plev", "pressure_level", "isobaricinhpa"):
            coord_rename[c] = "level"
    if coord_rename:
        ds = ds.rename(coord_rename)

    # Ensure longitude is 0-360 (ERA5 convention)
    if "longitude" in ds.coords:
        lon = ds.longitude.values
        if lon.min() < 0:
            ds = ds.assign_coords(longitude=(ds.longitude % 360))
            ds = ds.sortby("longitude")

    # Ensure latitude is descending (90 to -90, ERA5 convention)
    if "latitude" in ds.coords:
        if ds.latitude.values[0] < ds.latitude.values[-1]:
            ds = ds.isel(latitude=slice(None, None, -1))

    # Add a time coordinate matching ERA5 format
    if "time" not in ds.coords:
        target_dt = grib_file.stem.split("_")
        try:
            date_part = target_dt[2] if len(target_dt) > 2 else target_dt[-1]
            ds = ds.expand_dims("time").assign_coords(
                time=[pd.Timestamp(date_part[:8])])
        except Exception:
            ds = ds.expand_dims("time").assign_coords(
                time=[pd.Timestamp.now().floor("6h")])

    logger.success(
        f"ECMWF dataset ready | "
        f"dims={dict(ds.sizes)} | "
        f"vars={list(ds.data_vars)}")
    return ds


def load_realtime_init_state(
    cache_dir: str = "./cache",
    lag_hours: int = 6,
) -> Optional[tuple]:
    """
    High-level function: download latest ECMWF analysis and return
    as xarray Dataset in ERA5 format, ready for NeuralGCM regridding.

    This is the main function called by the pipeline.
    """
    logger.info("Loading real-time ECMWF atmospheric state...")

    grib_file = download_ecmwf_analysis(
        cache_dir=cache_dir,
        lag_hours=lag_hours,
    )
    if grib_file is None:
        logger.error("ECMWF download failed completely")
        return None

    ds = grib_to_xarray(grib_file)
    if ds is None:
        logger.error("GRIB conversion failed")
        return None

    init_time = get_latest_available_date(lag_hours)
    logger.success(
        f"Real-time init state loaded | "
        f"Valid time: {init_time.strftime('%Y-%m-%d %H:%M UTC')}")
    return ds, init_time
