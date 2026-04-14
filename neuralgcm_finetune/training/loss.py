"""
Regional Fine-Tuning Loss Functions
=====================================
Computes loss ONLY over the Indian subcontinent domain
(6N-37N, 68E-97E) since that is where your GRIB data exists.

This is critical for correctness: if we compute loss over the
full NeuralGCM 2.8deg global grid, areas outside your region have
no target data and would produce NaN or zero-gradient regions
that corrupt the parameter updates.

Loss components (paper Appendix G.4):
  Phase A: L = area_weighted_MSE(pred, truth) over India domain
  Phase B: L = lambda_data * L_data + lambda_spec * L_spec + lambda_bias * L_bias
               all computed over India domain only
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple
from functools import partial


def build_regional_mask(
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    lat_min: float = 6.0,
    lat_max: float = 37.0,
    lon_min: float = 68.0,
    lon_max: float = 97.0,
) -> np.ndarray:
    """
    Build boolean mask for the Indian subcontinent domain
    on the NeuralGCM 2.8deg Gaussian grid.

    model_lats: (nlat,) array of NeuralGCM latitude coords
    model_lons: (nlon,) array of NeuralGCM longitude coords

    Returns: (nlat, nlon) boolean mask, True = inside India domain
    """
    lons_360 = model_lons % 360
    lon_mask = ((lons_360 >= lon_min) & (lons_360 <= lon_max))
    lat_mask = ((model_lats >= lat_min) & (model_lats <= lat_max))
    # Outer product to get 2D mask
    mask_2d  = np.outer(lat_mask, lon_mask)
    n_points = mask_2d.sum()
    n_total  = mask_2d.size
    frac     = 100 * n_points / n_total
    from loguru import logger
    logger.info(
        f"Regional mask: {n_points}/{n_total} grid points "
        f"({frac:.1f}%) inside India domain "
        f"[{lat_min}N-{lat_max}N, {lon_min}E-{lon_max}E]")
    return mask_2d


def area_weights_regional(
    model_lats: np.ndarray,
    mask_2d: np.ndarray,
) -> np.ndarray:
    """
    Area weights restricted to regional mask.
    cos(lat) weighting, normalised to mean=1 over masked region.
    """
    lats_rad  = np.deg2rad(model_lats)
    cos_lats  = np.cos(lats_rad)
    # Broadcast to 2D
    w_2d      = np.outer(cos_lats, np.ones(mask_2d.shape[1]))
    # Zero out outside mask
    w_masked  = w_2d * mask_2d.astype(float)
    # Normalise
    mean_w    = w_masked[mask_2d].mean() if mask_2d.any() else 1.0
    if mean_w > 0:
        w_masked /= mean_w
    return w_masked


def mse_regional(
    pred:    jnp.ndarray,
    target:  jnp.ndarray,
    weights: jnp.ndarray,  # (nlat, nlon) area weights, 0 outside mask
) -> jnp.ndarray:
    """
    Area-weighted MSE over regional mask only.
    pred/target: (..., nlat, nlon)
    weights:     (nlat, nlon) -- zero outside India domain
    """
    diff = pred - target
    sq   = diff ** 2
    # Broadcast weights to match leading dims
    w = weights.reshape(
        *([1] * (sq.ndim - 2)), *weights.shape)
    # Sum over all dims, divide by number of valid points
    n_valid = jnp.maximum(jnp.sum(weights > 0), 1.0)
    # Average over spatial dims using weights, then mean over leading dims
    weighted_sq = sq * w
    spatial_sum = jnp.sum(weighted_sq, axis=(-2, -1))
    return jnp.mean(spatial_sum) / n_valid


def spectral_mse_regional(
    pred:    jnp.ndarray,
    target:  jnp.ndarray,
    mask:    jnp.ndarray,  # (nlat, nlon) boolean mask
    cutoff:  int = 15,     # lower cutoff for regional domain
) -> jnp.ndarray:
    """
    Spectral MSE over the regional domain.
    Applied after masking non-India points to zero.
    cutoff=15 appropriate for regional 2.8deg domain
    (vs cutoff=42 for global domain).
    """
    # Zero out non-India points before FFT
    m     = jnp.array(mask, dtype=jnp.float32)
    pred_m   = pred   * m
    target_m = target * m
    # 2D FFT over spatial dims
    pred_fft   = jnp.fft.rfft2(pred_m,   axes=(-2, -1))
    target_fft = jnp.fft.rfft2(target_m, axes=(-2, -1))
    pred_pow   = jnp.abs(pred_fft)   ** 2
    target_pow = jnp.abs(target_fft) ** 2
    # Compare power spectra up to cutoff wavenumber
    pred_spec   = pred_pow[...,   :cutoff]
    target_spec = target_pow[..., :cutoff]
    return jnp.mean((pred_spec - target_spec) ** 2)


# Rescaling factors (paper Appendix G.3)
RESCALE = {
    "temperature":                        1.0,
    "geopotential":                       0.5,
    "specific_humidity":                  0.66,
    "log_surface_pressure":               5.0,
    "u_component_of_wind":                1.0,
    "v_component_of_wind":                1.0,
    "specific_cloud_liquid_water_content": 0.05,
    "specific_cloud_ice_water_content":   0.05,
}


class RegionalFineTuneLoss:
    """
    Fine-tuning loss restricted to Indian subcontinent domain.
    Supports Phase A (MSE only) and Phase B (MSE+spectral+bias).
    """

    def __init__(
        self,
        config: dict,
        model_lats: np.ndarray,
        model_lons: np.ndarray,
    ):
        self.config = config
        region = config["data"]["region"]
        # Build regional mask on NeuralGCM 2.8deg Gaussian grid
        self.mask = build_regional_mask(
            model_lats, model_lons,
            lat_min=region["lat_min"],
            lat_max=region["lat_max"],
            lon_min=region["lon_min"],
            lon_max=region["lon_max"],
        )
        self.weights = area_weights_regional(
            model_lats, self.mask)
        # Convert to JAX arrays (immutable)
        self.mask_j    = jnp.array(self.mask)
        self.weights_j = jnp.array(self.weights)

    def compute(
        self,
        preds:   Dict[str, jnp.ndarray],
        targets: Dict[str, jnp.ndarray],
        lead_time_hours: int = 6,
        phase: str = "a",
        lambda_data: float = 15.0,
        lambda_spec: float = 0.05,
        lambda_bias: float = 1.5,
    ) -> Dict[str, jnp.ndarray]:
        """Compute regional fine-tuning loss."""
        total = jnp.zeros(())
        info  = {}

        common = [v for v in preds if v in targets]
        if not common:
            info["total"] = total
            return info

        for var in common:
            pred = preds[var]
            tgt  = targets[var]

            # Match spatial dims
            nlat = min(pred.shape[-2], tgt.shape[-2],
                       self.weights_j.shape[0])
            nlon = min(pred.shape[-1], tgt.shape[-1],
                       self.weights_j.shape[1])
            pred = pred[..., :nlat, :nlon]
            tgt  = tgt[...,  :nlat, :nlon]
            w    = self.weights_j[:nlat, :nlon]
            msk  = self.mask_j[:nlat, :nlon]

            # Rescale
            scale  = RESCALE.get(var, 1.0)
            tau_f  = (1.0 + lead_time_hours / 24.0) ** (-0.5)
            pred_r = pred * scale * tau_f
            tgt_r  = tgt  * scale * tau_f

            if phase == "a":
                lv = mse_regional(pred_r, tgt_r, w)
                total = total + lv
                info[f"mse_{var}"] = lv
            else:
                l_data = mse_regional(pred_r, tgt_r, w)
                l_spec = spectral_mse_regional(pred_r, tgt_r, msk)
                l_bias = mse_regional(
                    jnp.mean(pred_r, axis=0, keepdims=True),
                    jnp.mean(tgt_r,  axis=0, keepdims=True),
                    w)
                lv = (lambda_data * l_data +
                      lambda_spec * l_spec +
                      lambda_bias * l_bias)
                total = total + lv
                info[f"data_{var}"]  = l_data
                info[f"spec_{var}"]  = l_spec
                info[f"bias_{var}"]  = l_bias

        info["total"] = total
        return info
