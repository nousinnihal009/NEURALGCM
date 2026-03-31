import numpy as np
import xarray as xr
import logging

logger = logging.getLogger(__name__)

def align_spatiotemporal(fcst_ds: xr.Dataset, obs_ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Rigorously aligns two datasets in space and time.
    Observations (ERA5) are typically interpolated to match the forecast grid 
    if differing physically, or vice versa, to ensure apples-to-apples comparison.
    """
    logger.info("Performing spatiotemporal alignment...")
    
    # 1. Ensure they share the same spatial domain (bbox intersection)
    min_lat = max(float(fcst_ds.latitude.min()), float(obs_ds.latitude.min()))
    max_lat = min(float(fcst_ds.latitude.max()), float(obs_ds.latitude.max()))
    min_lon = max(float(fcst_ds.longitude.min()), float(obs_ds.longitude.min()))
    max_lon = min(float(fcst_ds.longitude.max()), float(obs_ds.longitude.max()))
    
    logger.info(f"Clipping to bounds: lat=[{min_lat}, {max_lat}], lon=[{min_lon}, {max_lon}]")
    
    fcst_ds = fcst_ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
    obs_ds = obs_ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
    
    # 2. Spatial Alignment (Interpolation)
    if not fcst_ds.latitude.equals(obs_ds.latitude) or not fcst_ds.longitude.equals(obs_ds.longitude):
        logger.info("Interpolating observations to forecast grid (bilinear)...")
        obs_ds = obs_ds.interp(
            latitude=fcst_ds.latitude, 
            longitude=fcst_ds.longitude, 
            method="bilinear"
        )
    
    # 3. Temporal Alignment
    if 'time' in fcst_ds.coords and 'time' in obs_ds.coords:
        # Find time intersection
        common_times = np.intersect1d(fcst_ds.time.values, obs_ds.time.values)
        if len(common_times) == 0:
            logger.warning("No exact time intersection! Relying on nearest-neighbor matching.")
            obs_ds = obs_ds.reindex(time=fcst_ds.time, method='nearest')
        else:
            fcst_ds = fcst_ds.sel(time=common_times)
            obs_ds = obs_ds.sel(time=common_times)
            
    logger.info("Alignment complete.")
    return fcst_ds, obs_ds
