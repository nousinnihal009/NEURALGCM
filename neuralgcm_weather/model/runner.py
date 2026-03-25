"""
NeuralGCM Forecast Runner
==========================
Clean, correct implementation of the NeuralGCM inference pipeline.
All bugs from forecast_anywhere.py are fixed here permanently.
"""

import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_FLAGS"]                     = "--xla_cpu_use_thunk_runtime=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
import numpy as np
import xarray as xr
import pandas as pd
import jax
import jax.numpy as jnp
from loguru import logger
from typing import Optional, Tuple
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils


def build_regridder(era5_dataset: xr.Dataset, model) -> object:
    """
    Build conservative regridder from ERA5 0.25deg to NeuralGCM 2.8deg.
    Called once and reused for all forecasts.
    """
    lat_coord = ("latitude" if "latitude" in era5_dataset.coords
                 else "lat")
    lon_coord = ("longitude" if "longitude" in era5_dataset.coords
                 else "lon")

    era5_grid = spherical_harmonic.Grid(
        latitude_nodes   = era5_dataset.sizes[lat_coord],
        longitude_nodes  = era5_dataset.sizes[lon_coord],
        latitude_spacing = xarray_utils.infer_latitude_spacing(
            era5_dataset[lat_coord]),
        longitude_offset = xarray_utils.infer_longitude_offset(
            era5_dataset[lon_coord]),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True)
    logger.info("Regridder built: ERA5 0.25deg -> NeuralGCM 2.8deg")
    return regridder


def regrid_init_state(
    era5_slice: xr.Dataset,
    regridder,
) -> xr.Dataset:
    """Regrid ERA5 slice to NeuralGCM Gaussian grid."""
    ev = xarray_utils.regrid(era5_slice, regridder)
    ev = xarray_utils.fill_nan_with_nearest(ev)
    logger.info(f"Regridded dims: {dict(ev.sizes)}")
    return ev


def run_forecast(
    model,
    init_state: xr.Dataset,
    forecast_days: int = 5,
    timestep_hours: int = 24,
    rng_seed: int = 0,
) -> Tuple[xr.Dataset, float]:
    """
    Run NeuralGCM forecast. Returns (predictions_dataset, elapsed_seconds).

    FIXED BUGS vs forecast_anywhere.py:
    - unroll() takes raw forcings, no expand_dims
    - steps=forecast_days (not +1)
    - data_to_xarray called with preds directly
    - proper handling of namedtuple vs dict preds
    """
    logger.info(
        f"Running {forecast_days}-day NeuralGCM forecast "
        f"(timestep={timestep_hours}h)...")
    t0 = time.time()

    # Prepare inputs
    inputs   = model.inputs_from_xarray(init_state)
    forcings = model.forcings_from_xarray(init_state)
    rng_key  = jax.random.key(rng_seed)

    # Encode initial state
    state = model.encode(inputs, forcings, rng_key)

    # CORRECT unroll call — no expand_dims, no steps+1
    _, preds = model.unroll(
        state,
        forcings,
        steps     = forecast_days,
        timedelta = np.timedelta64(timestep_hours, "h"),
        start_with_input = True,
    )

    elapsed = time.time() - t0
    logger.info(f"Unroll complete in {elapsed:.1f}s")

    # Convert to xarray — handle all preds container types
    n_steps = forecast_days + 1
    times_td = pd.to_timedelta(
        np.arange(n_steps) * timestep_hours, "h")

    ds = _safe_data_to_xarray(model, preds, times_td)

    logger.success(
        f"Forecast complete | "
        f"elapsed={elapsed:.1f}s | "
        f"vars={list(ds.data_vars)} | "
        f"dims={dict(ds.sizes)}")
    return ds, elapsed


def _safe_data_to_xarray(model, preds, times_td) -> xr.Dataset:
    """
    Convert preds (namedtuple/dict/object) to xarray Dataset.
    Handles sim_time stripping and all container types safely.
    """
    # Try direct call first (most common case)
    try:
        return model.data_to_xarray(preds, times=times_td)
    except Exception as e:
        logger.warning(f"data_to_xarray direct failed: {e}")

    # Try stripping sim_time
    try:
        if hasattr(preds, "_asdict"):
            preds_dict = {k: v for k, v in preds._asdict().items()
                          if k != "sim_time"}
        elif isinstance(preds, dict):
            preds_dict = {k: v for k, v in preds.items()
                          if k != "sim_time"}
        elif hasattr(preds, "__dict__"):
            preds_dict = {k: v for k, v in preds.__dict__.items()
                          if k != "sim_time"}
        else:
            raise ValueError(f"Unknown preds type: {type(preds)}")
        return model.data_to_xarray(preds_dict, times=times_td)
    except Exception as e:
        logger.error(f"data_to_xarray with strip also failed: {e}")
        raise
