"""
verification_metrics.py
========================
Deterministic and distribution-aware verification metrics for meteorological
forecast evaluation, following WMO / ECMWF standard verification practices.
"""
import numpy as np
import xarray as xr
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _weights(lats: xr.DataArray) -> xr.DataArray:
    """Latitude-based cosine weighting for area-weighted statistics."""
    w = np.cos(np.deg2rad(lats))
    return w / w.mean()


# ── Pointwise Deterministic Metrics ──────────────────────────────────────────

def mean_bias(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Mean Bias (forecast − observation), averaged over time."""
    return (fcst - obs).mean(dim='time')


def mae(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Mean Absolute Error, averaged over time."""
    return np.abs(fcst - obs).mean(dim='time')


def rmse(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Root Mean Square Error, averaged over time."""
    return np.sqrt(((fcst - obs) ** 2).mean(dim='time'))


def pearson_correlation(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Pearson correlation coefficient at each grid point over time."""
    return xr.corr(fcst, obs, dim='time')


# ── Spatial / Summary Metrics ────────────────────────────────────────────────

def spatial_rmse(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """
    Area-weighted RMSE collapsed over spatial dims, returned per timestep.
    Useful for tracking forecast skill degradation over time.
    """
    weights = _weights(fcst.latitude)
    sq_err = ((fcst - obs) ** 2)
    weighted = sq_err.weighted(weights)
    return np.sqrt(weighted.mean(dim=['latitude', 'longitude']))


def anomaly_correlation_coefficient(
    fcst: xr.DataArray,
    obs: xr.DataArray,
    clim: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """
    Anomaly Correlation Coefficient (ACC).
    If no climatology is provided, the temporal mean of observations is used.
    Returned per timestep (scalar per time).

    ACC = sum(w * f' * o') / sqrt(sum(w * f'^2) * sum(w * o'^2))
    where f' = fcst - clim, o' = obs - clim, w = cos(lat)
    """
    if clim is None:
        clim = obs.mean(dim='time')

    f_prime = fcst - clim
    o_prime = obs - clim

    weights = _weights(fcst.latitude)

    numerator = (weights * f_prime * o_prime).sum(dim=['latitude', 'longitude'])
    denom_f = (weights * f_prime ** 2).sum(dim=['latitude', 'longitude'])
    denom_o = (weights * o_prime ** 2).sum(dim=['latitude', 'longitude'])
    denominator = np.sqrt(denom_f * denom_o)

    acc = numerator / denominator.where(denominator > 0)
    return acc


def std_ratio(fcst: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Standard deviation ratio (forecast / obs) at each grid point over time."""
    return fcst.std(dim='time') / obs.std(dim='time').where(obs.std(dim='time') > 0)


# ── Aggregate Runner ─────────────────────────────────────────────────────────

def calculate_metrics(
    fcst_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    variables: list[str] | None = None
) -> Dict[str, Dict[str, xr.DataArray]]:
    """
    Compute all verification metrics for each shared variable.

    Returns
    -------
    dict : {variable_name: {metric_name: xr.DataArray}}
    """
    if variables is None:
        variables = list(set(fcst_ds.data_vars) & set(obs_ds.data_vars))

    if not variables:
        raise ValueError("No common variables found between forecast and observation datasets.")

    logger.info(f"Computing metrics for variables: {variables}")
    results: Dict[str, Dict[str, xr.DataArray]] = {}

    for var in variables:
        f, o = fcst_ds[var], obs_ds[var]
        logger.info(f"  → {var}")
        results[var] = {
            'bias': mean_bias(f, o),
            'mae': mae(f, o),
            'rmse': rmse(f, o),
            'pearson_r': pearson_correlation(f, o),
            'spatial_rmse': spatial_rmse(f, o),
            'acc': anomaly_correlation_coefficient(f, o),
            'std_ratio': std_ratio(f, o),
        }

    return results
