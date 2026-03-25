#!/usr/bin/env python3
"""
NeuralGCM GPU deterministic forecast - 5 days, 20 steps x 6h.
"""

import pickle
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax

t0_wall = time.time()

print("=" * 70)
print("  NeuralGCM GPU Deterministic Forecast - 5 days (20 steps x 6h)")
print("=" * 70)
print(f"  JAX version  : {jax.__version__}")
print(f"  JAX backend  : {jax.default_backend()}")
devices = jax.devices()
print(f"  Devices      : {devices}")
if devices and hasattr(devices[0], "device_kind"):
    gpu_name = getattr(devices[0], "device_kind", str(devices[0]))
    print(f"  GPU          : {gpu_name}")
else:
    gpu_name = str(devices[0]) if devices else "unknown"
print("=" * 70)

import gcsfs
import neuralgcm
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

# Config
MODEL_PATH = "v1/deterministic_2_8_deg.pkl"
ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
INIT_TIME = "2020-01-15T00:00"
OUTER_STEPS = 20
INNER_HOURS = 6
TIMEDELTA = np.timedelta64(INNER_HOURS, "h")

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
WHITE = "white"


def main():
    # 1. Load checkpoint
    print("\n[1/5] Loading checkpoint from GCS...")
    gcs = gcsfs.GCSFileSystem(token="anon")
    with gcs.open(f"gs://neuralgcm/models/{MODEL_PATH}", "rb") as f:
        ckpt = pickle.load(f)
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    print(f"      Loaded: {MODEL_PATH}")

    # 2. Open ARCO-ERA5
    print("\n[2/5] Opening ARCO-ERA5 on GCS...")
    full_era5 = xr.open_zarr(
        ERA5_ZARR,
        chunks=None,
        storage_options=dict(token="anon"),
    )
    needed = list(set(model.input_variables) | set(model.forcing_variables))
    init_dt = pd.Timestamp(INIT_TIME)
    sliced = (
        full_era5[needed]
        .sel(time=init_dt, method="nearest")
        .expand_dims("time")
        .compute()
    )
    print(f"      Init: {init_dt}")

    # 3 & 4. Regrid ERA5 → model grid
    print("\n[3/5] Regridding ERA5 (0.25 deg) -> model (2.8 deg)...")
    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=full_era5.sizes["latitude"],
        longitude_nodes=full_era5.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True
    )
    eval_era5 = xarray_utils.regrid(sliced, regridder)
    eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)
    print(f"      Shape: {dict(eval_era5.sizes)}")

    # 5. Run forecast
    print("\n[4/5] Running 5-day deterministic forecast (20 x 6h)...")
    t0_fc = time.time()
    inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
    forcings = model.forcings_from_xarray(eval_era5)
    rng = jax.random.key(0)
    state = model.encode(inputs, forcings, rng)
    _, preds = model.unroll(
        state,
        forcings,
        steps=OUTER_STEPS,
        timedelta=TIMEDELTA,
        start_with_input=True,
    )
    jax.block_until_ready(preds)
    elapsed_fc = time.time() - t0_fc
    print(f"      Forecast done in {elapsed_fc:.1f}s")

    times_h = list(range(0, OUTER_STEPS * INNER_HOURS, INNER_HOURS))
    times_td = pd.to_timedelta(times_h, "h")
    ds = model.data_to_xarray(preds, times=times_td)

    # Detect time coord
    tc = "prediction_timedelta" if "prediction_timedelta" in ds.coords else "time"

    # 6. Plots
    print("\n[5/5] Saving plots...")

    # a) Z500 2x3 grid (Day 0..5)
    zvar = "geopotential" if "geopotential" in ds else list(ds.data_vars)[0]
    lev = ds[zvar].coords.get("level", None)
    if lev is not None:
        idx500 = int(np.argmin(np.abs(lev.values - 500)))
    else:
        idx500 = 0

    fig_a, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=BG)
    axes = axes.flatten()
    day_indices = [0, 4, 8, 12, 16, min(19, ds.dims[tc] - 1)]  # Day 0..5
    for ax, di in zip(axes, day_indices):
        if lev is not None:
            data = ds[zvar].isel(**{tc: di}, level=idx500).values
        else:
            data = ds[zvar].isel(**{tc: di}).values.squeeze()
        ax.imshow(data, origin="upper", cmap="RdYlBu_r", aspect="auto", interpolation="bilinear")
        ax.set_facecolor(PANEL)
        ax.set_title(f"Day {di * INNER_HOURS // 24}", color=WHITE, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(BORDER)
    fig_a.suptitle("Z500 (m2/s2) - NeuralGCM 2.8 deg", color=WHITE, fontsize=14, y=1.02)
    plt.tight_layout()
    fig_a.savefig("forecast_z500.png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig_a)
    print("      forecast_z500.png")

    # b) Global-mean 850 hPa temperature
    tvar = next((v for v in ["temperature", "2m_temperature"] if v in ds.data_vars), None)
    if tvar:
        tdata = ds[tvar]
        dims_to_mean = [d for d in tdata.dims if d != tc]
        tseries = tdata.mean(dims_to_mean)
        t_label = "Global Mean " + tvar.replace("_", " ").title() + " (K)"
        hours = [t.total_seconds() / 3600 for t in pd.to_timedelta(tseries[tc].values)]
        fig_b, ax_b = plt.subplots(figsize=(10, 4), facecolor=BG)
        ax_b.set_facecolor(PANEL)
        ax_b.plot(hours, tseries.values, color="#58a6ff", lw=2, marker="o", ms=4)
        ax_b.set_xlabel("Lead time (h)", color=WHITE)
        ax_b.set_ylabel(t_label, color=WHITE)
        ax_b.set_title("Global Mean 850 hPa Temperature", color=WHITE, fontsize=12)
        ax_b.tick_params(colors=WHITE)
        ax_b.grid(True, color=BORDER, ls="--", alpha=0.6)
        for s in ax_b.spines.values():
            s.set_edgecolor(BORDER)
        plt.tight_layout()
        fig_b.savefig("forecast_temp.png", dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close(fig_b)
        print("      forecast_temp.png")

    # c) Global-mean MSLP (surface pressure)
    spvar = next((v for v in ["surface_pressure", "log_surface_pressure", "sp"] if v in ds.data_vars), None)
    if spvar:
        sp = ds[spvar]
        spseries = sp.mean([d for d in sp.dims if d != tc and d in sp.dims])
        hours = [t.total_seconds() / 3600 for t in pd.to_timedelta(spseries[tc].values)]
        fig_c, ax_c = plt.subplots(figsize=(10, 4), facecolor=BG)
        ax_c.set_facecolor(PANEL)
        ax_c.plot(hours, spseries.values, color="#7ee787", lw=2, marker="o", ms=4)
        ax_c.set_xlabel("Lead time (h)", color=WHITE)
        ax_c.set_ylabel("Global mean " + spvar + " (Pa)", color=WHITE)
        ax_c.set_title("Global Mean MSLP (surface pressure)", color=WHITE, fontsize=12)
        ax_c.tick_params(colors=WHITE)
        ax_c.grid(True, color=BORDER, ls="--", alpha=0.6)
        for s in ax_c.spines.values():
            s.set_edgecolor(BORDER)
        plt.tight_layout()
        fig_c.savefig("forecast_mslp.png", dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close(fig_c)
        print("      forecast_mslp.png")
    else:
        print("      (skipping forecast_mslp.png: no surface_pressure var)")

    # d) Zonal-mean wind by level (horizontal bar)
    uvar = next((v for v in ["u_component_of_wind", "u_wind", "u"] if v in ds.data_vars), None)
    if uvar and "level" in ds[uvar].dims:
        udata = ds[uvar]
        if "longitude" in udata.dims:
            uzon = udata.isel(**{tc: -1}).mean("longitude")
        elif "lon" in udata.dims:
            uzon = udata.isel(**{tc: -1}).mean("lon")
        else:
            uzon = udata.isel(**{tc: -1})
        other_dims = [d for d in uzon.dims if d != "level"]
        uglob = uzon.mean(other_dims) if other_dims else uzon
        levels = uglob.level.values
        fig_d, ax_d = plt.subplots(figsize=(8, 6), facecolor=BG)
        ax_d.set_facecolor(PANEL)
        ax_d.barh(levels, uglob.values, color="#f78166", alpha=0.8, height=12)
        ax_d.set_xlabel("Zonal-mean u (m/s)", color=WHITE)
        ax_d.set_ylabel("Pressure (hPa)", color=WHITE)
        ax_d.set_title("Zonal-Mean Wind by Level (final step)", color=WHITE, fontsize=12)
        ax_d.tick_params(colors=WHITE)
        ax_d.invert_yaxis()
        for s in ax_d.spines.values():
            s.set_edgecolor(BORDER)
        plt.tight_layout()
        fig_d.savefig("forecast_wind.png", dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close(fig_d)
        print("      forecast_wind.png")
    else:
        print("      (skipping forecast_wind.png: no u_wind or level)")

    elapsed = time.time() - t0_wall
    print("\n" + "=" * 70)
    print("  DONE")
    print(f"  GPU          : {gpu_name}")
    print(f"  Wall time    : {elapsed:.1f}s")
    print(f"  JAX backend  : {jax.default_backend()}")
    print(f"  Output shape : {dict(ds.dims)}")
    print("  Saved: forecast_z500.png, forecast_temp.png, forecast_mslp.png, forecast_wind.png")
    print("=" * 70)
    return ds


if __name__ == "__main__":
    ds = main()
    print("\nds.dims:", dict(ds.dims))
    print("ds.data_vars:", list(ds.data_vars))
