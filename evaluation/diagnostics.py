"""
diagnostics.py
===============
Advanced diagnostic analyses: error structure, temporal degradation,
physical consistency checks, and regional performance segmentation.
"""
import numpy as np
import xarray as xr
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Error Structure Analysis ─────────────────────────────────────────────────

def spatial_error_decomposition(
    fcst: xr.DataArray,
    obs: xr.DataArray
) -> Dict[str, xr.DataArray]:
    """
    Decompose the total MSE into systematic (bias²) and random components.
    MSE = Bias² + Variance_error
    """
    diff = fcst - obs
    bias = diff.mean(dim='time')
    variance_err = diff.var(dim='time')
    mse = (diff ** 2).mean(dim='time')

    return {
        'mse': mse,
        'bias_squared': bias ** 2,
        'variance_error': variance_err,
        'systematic_fraction': (bias ** 2) / mse.where(mse > 0),
    }


def detect_systematic_bias(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    threshold_std: float = 1.5
) -> xr.DataArray:
    """
    Flag grid points where systematic bias exceeds threshold_std * spatial_std(bias).
    Returns a boolean mask of flagged points.
    """
    bias = (fcst - obs).mean(dim='time')
    bias_std = float(bias.std())
    flagged = np.abs(bias) > threshold_std * bias_std
    n_flagged = int(flagged.sum())
    logger.info(f"Systematic bias detection: {n_flagged} grid points flagged "
                f"(>{threshold_std}σ threshold).")
    return flagged


# ── Temporal Degradation ─────────────────────────────────────────────────────

def temporal_error_evolution(
    fcst: xr.DataArray,
    obs: xr.DataArray
) -> xr.DataArray:
    """
    Compute RMSE at each timestep (collapsed over spatial dims).
    This reveals forecast skill degradation over time / lead time.
    """
    weights = np.cos(np.deg2rad(fcst.latitude))
    sq_err = ((fcst - obs) ** 2)
    weighted = sq_err.weighted(weights)
    return np.sqrt(weighted.mean(dim=['latitude', 'longitude']))


def bias_drift(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    window: int = 6
) -> xr.DataArray:
    """
    Rolling-mean bias over a temporal window (default 6 timesteps).
    Useful for detecting slow bias drift over time.
    """
    bias_ts = (fcst - obs).mean(dim=['latitude', 'longitude'])
    return bias_ts.rolling(time=window, center=True).mean()


# ── Physical Consistency Checks ──────────────────────────────────────────────

def wind_speed_consistency(ds: xr.Dataset) -> Optional[xr.DataArray]:
    """
    Cross-variable check: compute wind speed from u10 / v10 components
    and flag implausible values (> 60 m/s at 10 m).
    """
    u_name = _find_var(ds, ['u_wind_10m', 'u10'])
    v_name = _find_var(ds, ['v_wind_10m', 'v10'])
    if u_name is None or v_name is None:
        logger.info("Wind components not found; skipping consistency check.")
        return None

    ws = np.sqrt(ds[u_name] ** 2 + ds[v_name] ** 2)
    implausible = ws > 60.0
    n_imp = int(implausible.sum())
    if n_imp > 0:
        logger.warning(f"Physical inconsistency: {n_imp} wind speed values > 60 m/s.")
    return implausible


def temperature_range_check(
    ds: xr.Dataset,
    min_k: float = 180.0,
    max_k: float = 340.0
) -> Optional[xr.DataArray]:
    """Flag temperatures outside physically plausible range [180, 340] K."""
    t_name = _find_var(ds, ['temperature_2m', 't2m'])
    if t_name is None:
        return None

    t = ds[t_name]
    # If already in Celsius, adjust bounds
    if t.attrs.get('units', 'K') == 'C':
        min_val, max_val = min_k - 273.15, max_k - 273.15
    else:
        min_val, max_val = min_k, max_k

    out_of_range = (t < min_val) | (t > max_val)
    n = int(out_of_range.sum())
    if n > 0:
        logger.warning(f"Temperature plausibility: {n} values outside [{min_val:.1f}, {max_val:.1f}].")
    return out_of_range


# ── Regional Performance Segmentation ────────────────────────────────────────

def regional_metrics(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    land_mask: Optional[xr.DataArray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Segment performance by latitude band (tropical / mid-latitude / polar).
    Returns dict of {region: {metric: value}}.
    """
    lat = fcst.latitude
    regions = {
        'tropical': (lat >= -23.5) & (lat <= 23.5),
        'northern_midlat': (lat > 23.5) & (lat <= 60),
        'southern_midlat': (lat < -23.5) & (lat >= -60),
        'polar': (np.abs(lat) > 60),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, mask in regions.items():
        if not mask.any():
            continue
        f_reg = fcst.sel(latitude=lat[mask])
        o_reg = obs.sel(latitude=lat[mask])
        diff = f_reg - o_reg
        results[name] = {
            'bias': float(diff.mean()),
            'rmse': float(np.sqrt((diff ** 2).mean())),
            'mae': float(np.abs(diff).mean()),
        }
        logger.info(f"  Region '{name}': RMSE={results[name]['rmse']:.4f}")

    return results


# ── Aggregate Diagnostics Runner ─────────────────────────────────────────────

def compute_diagnostics(
    fcst_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    variables: list[str] | None = None,
) -> Dict[str, dict]:
    """
    Run all diagnostic analyses on each shared variable.

    Returns
    -------
    dict : {variable: {diagnostic_name: result}}
    """
    if variables is None:
        variables = list(set(fcst_ds.data_vars) & set(obs_ds.data_vars))

    logger.info(f"Running diagnostics for: {variables}")
    diag: Dict[str, dict] = {}

    for var in variables:
        f, o = fcst_ds[var], obs_ds[var]
        diag[var] = {
            'error_decomposition': spatial_error_decomposition(f, o),
            'systematic_bias_flags': detect_systematic_bias(f, o),
            'temporal_rmse': temporal_error_evolution(f, o),
            'bias_drift': bias_drift(f, o),
            'regional': regional_metrics(f, o),
        }

    # Cross-variable physical checks (on forecast dataset)
    diag['_physical_checks'] = {
        'wind_implausible': wind_speed_consistency(fcst_ds),
        'temperature_implausible': temperature_range_check(fcst_ds),
    }

    return diag


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_var(ds: xr.Dataset, candidates: list[str]) -> Optional[str]:
    """Return the first variable name from candidates that exists in ds."""
    for c in candidates:
        if c in ds.data_vars:
            return c
    return None
