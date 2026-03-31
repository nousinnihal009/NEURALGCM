import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger(__name__)

VARIABLE_MAP = {
    'tp': 'total_precipitation',
    't2m': 'temperature_2m',
    'u10': 'u_wind_10m',
    'v10': 'v_wind_10m',
    'msl': 'mean_sea_level_pressure',
    'z': 'geopotential',
    'd2m': 'dewpoint_temperature_2m',
}

def standardize_names(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    for var in ds.data_vars:
        lower_var = str(var).lower()
        if lower_var in VARIABLE_MAP:
            rename_dict[var] = VARIABLE_MAP[lower_var]
            
    if rename_dict:
        logger.info(f"Standardizing variables: {rename_dict}")
        ds = ds.rename(rename_dict)
    return ds

def standardize_units(ds: xr.Dataset) -> xr.Dataset:
    ds_converted = ds.copy()
    
    for var in ds_converted.data_vars:
        attrs = ds_converted[var].attrs
        unit = attrs.get('units', '').lower()
        
        if unit == 'k' and 'temperature' in str(var).lower():
            logger.info(f"Converting {var} from K to C")
            ds_converted[var] = ds_converted[var] - 273.15
            ds_converted[var].attrs['units'] = 'C'
            
        elif unit == 'm' and 'precipitation' in str(var).lower():
            logger.info(f"Converting {var} from m to mm")
            ds_converted[var] = ds_converted[var] * 1000.0
            ds_converted[var].attrs['units'] = 'mm'
            
    return ds_converted

def standardize_coordinates(ds: xr.Dataset) -> xr.Dataset:
    rename_coords = {}
    for c in ds.coords:
        if str(c).lower() in ['lat', 'latitude']:
            rename_coords[c] = 'latitude'
        elif str(c).lower() in ['lon', 'longitude']:
            rename_coords[c] = 'longitude'
            
    ds = ds.rename(rename_coords)
    
    if 'latitude' in ds.coords and ds.latitude[0] < ds.latitude[-1]: 
        ds = ds.sortby('latitude', ascending=False)
        
    if 'longitude' in ds.coords and float(ds.longitude.max()) > 180:
        logger.info("Shifting longitude [0, 360] -> [-180, 180]")
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby('longitude')
        
    return ds

def standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    logger.info("Standardizing names, coords, and units.")
    ds = standardize_names(ds)
    ds = standardize_coordinates(ds)
    ds = standardize_units(ds)
    return ds
