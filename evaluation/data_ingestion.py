import xarray as xr
import cfgrib
import logging
import os
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)

def load_era5_robustly(file_path: str) -> xr.Dataset:
    """
    Robustly loads ERA5 GRIB files that map to different dimension topologies
    (e.g. step vs time for instantaneous vs accumulated parameters).
    Returns a unified xarray Dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERA5 file not found: {file_path}")

    logger.info(f"Loading ERA5 from {file_path}")
    
    try:
        # returns list of homogeneous datasets
        datasets = cfgrib.open_datasets(file_path)
    except Exception as e:
        logger.error(f"Failed to read GRIB with cfgrib.open_datasets: {e}")
        raise

    logger.info(f"Found {len(datasets)} sub-datasets with distinct topologies.")
    
    standardized = []
    for ds in datasets:
        # Standardize step dimension into valid_time if it exists
        if 'step' in ds.dims and 'valid_time' in ds.coords:
            if ds['valid_time'].ndim == 1:
                ds = ds.swap_dims({'step': 'valid_time'})
                ds = ds.rename({'valid_time': 'time'})
                if 'time' in ds.coords:
                    ds = ds.drop_vars('time') # Drop the scalar initialization time
        standardized.append(ds)

    try:
        merged = xr.merge(standardized, compat='override')
        logger.info("Successfully merged diverse GRIB subsets.")
        return merged
    except xr.MergeError as e:
        logger.warning(f"Merge conflict: {e}. Attempting dataset interpolation...")
        # Fallback loop to reindex all datasets to the main 'time' coordinate of the first one
        base_time = standardized[0].time
        reindexed = [ds.reindex(time=base_time, method='nearest') if 'time' in ds.dims else ds for ds in standardized]
        return xr.merge(reindexed, compat='override')


def load_forecast(file_path: str, fmt: str = 'netcdf') -> xr.Dataset:
    """Loads forecast output from various formats."""
    logger.info(f"Loading forecast: {file_path} (format: {fmt})")
    fmt = fmt.lower()
    if fmt in ['nc', 'netcdf']:
        return xr.open_dataset(file_path)
    elif fmt in ['grib', 'grib2']:
        return xr.open_dataset(file_path, engine='cfgrib')
    elif fmt == 'zarr':
        return xr.open_zarr(file_path)
    else:
        raise ValueError(f"Unsupported forecast format: {fmt}")
