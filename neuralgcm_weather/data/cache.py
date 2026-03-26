"""
Init State Disk Cache
======================
Caches regridded ERA5/ECMWF init states to disk as NetCDF.
Cache key = (source, init_date_str, model_checkpoint).
Prevents re-downloading and re-regridding the same data.
"""

import hashlib
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional


def _cache_key(source: str, init_time: pd.Timestamp,
               checkpoint: str) -> str:
    raw = f"{source}|{init_time.isoformat()}|{checkpoint}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def cache_path(cache_dir: str, source: str,
               init_time: pd.Timestamp,
               checkpoint: str) -> Path:
    key = _cache_key(source, init_time, checkpoint)
    return Path(cache_dir) / "init_states" / f"{key}.nc"


def load_cached_init_state(
    cache_dir: str,
    source: str,
    init_time: pd.Timestamp,
    checkpoint: str,
) -> Optional[xr.Dataset]:
    """Return cached regridded init state, or None if not cached."""
    p = cache_path(cache_dir, source, init_time, checkpoint)
    if p.exists():
        try:
            ds = xr.open_dataset(str(p))
            logger.info(
                f"Init state cache HIT | "
                f"source={source} | time={init_time} | {p.name}")
            return ds
        except Exception as e:
            logger.warning(f"Cache read failed ({p}): {e}")
            p.unlink(missing_ok=True)
    return None


def save_init_state_to_cache(
    ds: xr.Dataset,
    cache_dir: str,
    source: str,
    init_time: pd.Timestamp,
    checkpoint: str,
) -> None:
    """Save regridded init state to disk cache."""
    p = cache_path(cache_dir, source, init_time, checkpoint)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.to_netcdf(str(p))
        size_mb = p.stat().st_size / 1e6
        logger.success(
            f"Init state cached ({size_mb:.1f} MB) | "
            f"source={source} | time={init_time} | {p.name}")
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


def clear_old_cache(cache_dir: str, max_age_days: int = 7) -> int:
    """Delete cached files older than max_age_days. Returns count deleted."""
    cache_root = Path(cache_dir) / "init_states"
    if not cache_root.exists():
        return 0
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
    deleted = 0
    for f in cache_root.glob("*.nc"):
        mtime = pd.Timestamp(f.stat().st_mtime, unit="s")
        if mtime < cutoff:
            f.unlink()
            deleted += 1
    if deleted:
        logger.info(f"Cache cleaned: {deleted} files older than "
                    f"{max_age_days} days deleted")
    return deleted
