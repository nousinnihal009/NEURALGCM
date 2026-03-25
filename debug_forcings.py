import jax, jax.numpy as jnp, pickle, gcsfs, neuralgcm, pandas as pd, xarray as xr, time
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

MODEL_NAME = "v1/deterministic_2_8_deg.pkl"
ERA5_ZARR  = ("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")
gcs = gcsfs.GCSFileSystem(token="anon")
with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
    ckpt = pickle.load(f)
model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

era5 = xr.open_zarr(ERA5_ZARR, chunks=None, storage_options=dict(token="anon"))
ERA5_LAT = "latitude"
ERA5_LON = "longitude"
needed = list(set(model.input_variables) | set(model.forcing_variables))
era5_grid = spherical_harmonic.Grid(
    latitude_nodes   = era5.sizes[ERA5_LAT],
    longitude_nodes  = era5.sizes[ERA5_LON],
    latitude_spacing = xarray_utils.infer_latitude_spacing(era5[ERA5_LAT]),
    longitude_offset = xarray_utils.infer_longitude_offset(era5[ERA5_LON]),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True)

era5_slice = era5[needed].sel(time=pd.Timestamp("2020-06-01T00:00"), method="nearest").compute()
ev = xarray_utils.regrid(era5_slice, regridder)
ev = xarray_utils.fill_nan_with_nearest(ev)

forcings = model.forcings_from_xarray(ev)
print("Raw forcings shape:", jax.tree.map(lambda x: getattr(x, 'shape', 'NO_SHAPE'), forcings))
print("Raw forcings type:", type(forcings))
