import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — safe for Cursor terminal
import matplotlib.pyplot as plt

import jax
print(f"\n{'='*60}")
print(f"  JAX version : {jax.__version__}")
print(f"  Backend     : {jax.default_backend()}")
print(f"  Devices     : {jax.devices()}")
print(f"{'='*60}\n")

import gcsfs
import neuralgcm
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG 
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME   = "v1/deterministic_2_8_deg.pkl"  # laptop-safe default
INIT_TIME    = "2020-01-15T00:00"              # ERA5 initialisation time
INNER_STEPS  = 24                              # model hours between saved snapshots
OUTER_STEPS  = 5                               # number of snapshots → 5 days total
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PRETRAINED CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
print(f"[1/5] Loading checkpoint: {MODEL_NAME}")
print("      First run ~200 MB from GCS — takes 1-2 min on typical broadband\n")

try:
    gcs = gcsfs.GCSFileSystem(token="anon")
    with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
        ckpt = pickle.load(f)
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    print(f"  ✅ Checkpoint loaded")
    print(f"     Input variables   : {model.input_variables}")
    print(f"     Forcing variables : {model.forcing_variables}\n")
except MemoryError:
    print("  ❌ Out of memory loading checkpoint.")
    print("     → Already using the smallest model (2.8 deg). Try closing other apps.")
    raise
except Exception as e:
    print(f"  ❌ Checkpoint load failed: {e}")
    print("     → Check internet connection. GCS requires anonymous streaming.")
    raise


# ─────────────────────────────────────────────────────────────────────────────
# 2. STREAM ERA5 SLICE  (ARCO-ERA5, anonymous, no local download needed)
# ─────────────────────────────────────────────────────────────────────────────
print("[2/5] Opening ARCO-ERA5 on GCS (anonymous, no download)...")

ERA5_ZARR = (
    "gs://gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)

try:
    full_era5 = xr.open_zarr(
        ERA5_ZARR,
        chunks=None,
        storage_options=dict(token="anon"),
    )
    print(f"  ✅ Dataset opened  (dims: {dict(full_era5.sizes)})")
except Exception as e:
    print(f"  ❌ ERA5 open failed: {e}")
    print("     → Stable internet required for GCS streaming.")
    raise

# Select only the variables the model needs and load into memory
needed_vars = list(set(model.input_variables) | set(model.forcing_variables))
init_dt     = pd.Timestamp(INIT_TIME)

print(f"\n  Selecting variables : {needed_vars}")
print(f"  Initialisation time : {init_dt}\n")

sliced_era5 = (
    full_era5[needed_vars]
    .sel(time=init_dt, method="nearest")
    .expand_dims("time")  # keep time dim for model.unroll() forcings (needs 1D sim_time)
    .compute()            # pull one time-step into RAM
)
print(f"  ✅ ERA5 slice loaded: {dict(sliced_era5.sizes)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGRID ERA5 → MODEL GRID
#    NeuralGCM 2.8 deg uses a Gaussian grid; ERA5 is 0.25 deg regular lat/lon.
# ─────────────────────────────────────────────────────────────────────────────
print("[3/5] Regridding ERA5 (0.25 deg) → NeuralGCM grid (2.8 deg)...")

era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes["latitude"],
    longitude_nodes=full_era5.sizes["longitude"],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)

regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)

eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)
print(f"  ✅ Regrid complete: {dict(eval_era5.sizes)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. INITIALISE AND RUN FORECAST
# ─────────────────────────────────────────────────────────────────────────────
print("[4/5] Initialising model state...")

inputs          = model.inputs_from_xarray(eval_era5)
input_forcings  = model.forcings_from_xarray(eval_era5)
rng_key         = jax.random.key(42)   # deterministic models ignore this

initial_state = model.encode(inputs, input_forcings, rng_key)

timedelta = np.timedelta64(INNER_STEPS, "h")
times     = np.arange(OUTER_STEPS) * INNER_STEPS   # hours axis

print(f"  ✅ State initialised")
print(f"\n  Running {OUTER_STEPS}-step forecast "
      f"(each step = {INNER_STEPS}h  →  total {OUTER_STEPS * INNER_STEPS}h / "
      f"{OUTER_STEPS * INNER_STEPS // 24} days)...")

t0 = time.time()

try:
    _, predictions = model.unroll(
        initial_state,
        model.forcings_from_xarray(eval_era5),
        steps=OUTER_STEPS,
        timedelta=timedelta,
        start_with_input=True,
    )
except Exception as e:
    msg = str(e).lower()
    if "out of memory" in msg or "oom" in msg or "memory" in msg:
        print("  ❌ Out of Memory (OOM).")
        print(f"     → Try reducing OUTER_STEPS from {OUTER_STEPS} to 2 or 3.")
        print("     → Close browser tabs and other heavy apps.")
    else:
        print(f"  ❌ Forecast failed: {e}")
    raise

elapsed = time.time() - t0
print(f"\n  ✅ Forecast complete in {elapsed:.1f}s  "
      f"({elapsed / (OUTER_STEPS * INNER_STEPS):.2f}s per simulated hour)")

# Convert to xarray Dataset
predictions_ds = model.data_to_xarray(predictions, times=pd.to_timedelta(times, "h"))
print(f"  Output variables : {list(predictions_ds.data_vars)}")

# Detect time coordinate name (neuralgcm may use 'time' or 'prediction_timedelta')
TIME_COORD = "prediction_timedelta" if "prediction_timedelta" in predictions_ds.coords else "time"


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Generating plots...")

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
WHITE = "white"

# ── 5a. 500 hPa Geopotential height  (Day 0 vs final forecast day) ───────────
z_var = "geopotential" if "geopotential" in predictions_ds else list(predictions_ds.data_vars)[0]
levels_coord = predictions_ds[z_var].coords.get("level", None)

if levels_coord is not None:
    lev_idx   = int(np.argmin(np.abs(levels_coord.values - 500)))
    z_day0    = predictions_ds[z_var].isel(**{TIME_COORD: 0},  level=lev_idx).values
    z_dayN    = predictions_ds[z_var].isel(**{TIME_COORD: -1}, level=lev_idx).values
    lev_label = f"{int(levels_coord.values[lev_idx])} hPa"
else:
    z_day0    = predictions_ds[z_var].isel(**{TIME_COORD: 0}).values.squeeze()
    z_dayN    = predictions_ds[z_var].isel(**{TIME_COORD: -1}).values.squeeze()
    lev_label = ""

vmin = min(z_day0.min(), z_dayN.min())
vmax = max(z_day0.max(), z_dayN.max())

fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=BG,
                          gridspec_kw=dict(wspace=0.06))

end_dt = init_dt + pd.Timedelta(hours=OUTER_STEPS * INNER_STEPS)
labels = [f"T+0  |  {init_dt.date()}", f"T+{OUTER_STEPS*INNER_STEPS}h  |  {end_dt.date()}"]

for ax, data, label in zip(axes, [z_day0, z_dayN], labels):
    im = ax.imshow(data, origin="upper", cmap="RdYlBu_r",
                   aspect="auto", interpolation="bilinear",
                   vmin=vmin, vmax=vmax)
    ax.set_facecolor(PANEL)
    ax.set_title(label, color=WHITE, fontsize=13, pad=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                     fraction=0.04, pad=0.05, shrink=0.55)
cbar.set_label(
    f"{z_var.replace('_',' ').title()}  {lev_label}  (m²/s²)",
    color=WHITE, fontsize=11,
)
cbar.ax.xaxis.set_tick_params(color=WHITE)
plt.setp(cbar.ax.xaxis.get_ticklabels(), color=WHITE)

fig.suptitle(
    f"NeuralGCM 2.8°  ·  Deterministic  ·  {lev_label} "
    f"{z_var.replace('_',' ').title()}",
    color=WHITE, fontsize=15, fontweight="bold", y=1.02,
)

plt.savefig("forecast_500hPa.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("  Saved: forecast_500hPa.png")
plt.close()


# ── 5b. Global-mean temperature time series ──────────────────────────────────
t_var = next(
    (v for v in ["temperature", "2m_temperature", "t"] if v in predictions_ds),
    None,
)

if t_var is not None:
    t_data = predictions_ds[t_var]
    if "level" in t_data.dims:
        lev850 = int(np.argmin(np.abs(t_data.coords["level"].values - 850)))
        t_series = t_data.isel(level=lev850).mean(
            [d for d in t_data.dims if d not in (TIME_COORD,)])
        t_label = "850 hPa Temperature (K)"
    else:
        t_series = t_data.mean(
            [d for d in t_data.dims if d != TIME_COORD])
        t_label = f"{t_var.replace('_',' ').title()} (K)"

    t_hours = np.array([
        td.total_seconds() / 3600
        for td in pd.to_timedelta(t_series[TIME_COORD].values)
    ])
    t_vals = t_series.values

    fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax2.set_facecolor(PANEL)
    ax2.plot(t_hours, t_vals, color="#58a6ff", linewidth=2.5,
             marker="o", markersize=7, markerfacecolor="#f78166")
    ax2.set_xlabel("Forecast lead time (hours)", color=WHITE, fontsize=11)
    ax2.set_ylabel(t_label, color=WHITE, fontsize=11)
    ax2.set_title(
        f"NeuralGCM 2.8°  ·  Global Mean {t_var}  ·  Init {init_dt.date()}",
        color=WHITE, fontsize=13, fontweight="bold",
    )
    ax2.tick_params(colors=WHITE)
    ax2.grid(True, color=BORDER, linestyle="--", alpha=0.6)
    for spine in ax2.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    plt.savefig("forecast_temp.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("  Saved: forecast_temp.png")
    plt.close()
else:
    print(f"  No temperature variable found; skipping. "
          f"Available: {list(predictions_ds.data_vars)}")


# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
  DONE

  Model     : {MODEL_NAME}
  Init time : {init_dt}
  Forecast  : {OUTER_STEPS} x {INNER_STEPS}h = {OUTER_STEPS * INNER_STEPS}h total
  Wall time : {elapsed:.1f}s
  JAX device: {jax.default_backend()}

  Outputs:
    forecast_500hPa.png
    forecast_temp.png
{'='*60}
""")
