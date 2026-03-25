# ── JAX FIRST (Windows DLL race fix) ─────────────────────────
import os, sys
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_FLAGS"]                     = "--xla_cpu_use_thunk_runtime=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"]          = "3"
sys.stdout.reconfigure(encoding="utf-8")

import jax
import jax.numpy as jnp
_devices = jax.devices()

import time, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import gcsfs
from dinosaur import (horizontal_interpolation,
                      spherical_harmonic, xarray_utils)
import neuralgcm

print("=" * 70)
print(f"  NeuralGCM Universal Weather Forecast")
print(f"  JAX {jax.__version__}  |  backend: {jax.default_backend()}")
print("=" * 70)

# ═══════════════════════════════════════════════════════
#  USER INPUT — change all 5 lines
# ═══════════════════════════════════════════════════════
LOCATION_NAME = "Chennai, India"
LAT           =  13.0827
LON           =  80.2707

# INIT_DATE: the date you want the forecast to START from.
# ERA5 historical data available: 1979-01-01 to 2020-12-31
# For real-time 2026 forecasts: see OPTION B below
# Format: "YYYY-MM-DDTHH:MM"  always use T00:00 for midnight
INIT_DATE     = "2020-03-25T00:00"   # <<< CHANGE THIS DATE

FORECAST_DAYS = 5   # How many days ahead (1-10)
# ═══════════════════════════════════════════════════════

# ── Forecast Mode ─────────────────────────────────────
FORECAST_MODE = "historical"  # "historical" or "realtime"

# For historical mode: use this ERA5 date
HISTORICAL_ANALOG_DATE = "2020-03-25T00:00"

# For realtime mode: downloads today's ECMWF open data
# (requires internet, downloads ~500MB of GRIB files)
TODAY = "2026-03-25"   # tomorrow's forecast start


# ── ECMWF Downloader Functions (for realtime mode) ────
def download_ecmwf_init(target_date_str):
    """
    Download ECMWF operational analysis for a specific date.
    Returns path to downloaded GRIB file.
    ECMWF open data: ERA5-like fields, same variables.
    """
    import urllib.request, json

    date = pd.Timestamp(target_date_str)
    date_str = date.strftime("%Y%m%d")

    # ECMWF open data base URL (public, no auth needed)
    base = "https://data.ecmwf.int/forecasts"

    # Required pressure-level variables matching NeuralGCM inputs
    variables = [
        "u", "v", "t", "q",           # wind, temp, humidity
        "z",                            # geopotential
        "clwc", "ciwc",               # cloud water/ice
        "sp",                           # surface pressure
    ]
    levels = "1/2/3/5/7/10/20/30/50/70/100/150/200/" \
             "250/300/400/500/600/700/850/925/1000"

    grib_file = f"ecmwf_init_{date_str}.grib2"

    if os.path.exists(grib_file):
        print(f"  Using cached: {grib_file}")
        return grib_file

    # Try ECMWF open data API
    try:
        import ecmwf.opendata
        client = ecmwf.opendata.Client("ecmwf")
        client.retrieve(
            date=date_str,
            time="00",
            step=0,
            stream="oper",
            type="an",
            param=variables,
            levtype="pl",
            levelist=levels,
            target=grib_file,
        )
        print(f"  Downloaded: {grib_file}")
        return grib_file
    except Exception as e:
        print(f"  ECMWF download failed: {e}")
        print(f"  Install: pip install ecmwf-opendata cfgrib")
        return None


def grib_to_era5_format(grib_file):
    """Convert GRIB2 download to xarray matching ERA5 format."""
    try:
        import cfgrib
        ds_grib = xr.open_dataset(
            grib_file,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}
        )
        # Rename to match ERA5 variable names
        rename_map = {
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "t": "temperature",
            "q": "specific_humidity",
            "z": "geopotential",
            "clwc": "specific_cloud_liquid_water_content",
            "ciwc": "specific_cloud_ice_water_content",
        }
        ds_grib = ds_grib.rename({k:v for k,v in rename_map.items()
                                   if k in ds_grib})
        return ds_grib
    except Exception as e:
        print(f"  GRIB conversion failed: {e}")
        return None


def install_realtime_deps():
    """Install required packages for realtime mode."""
    import subprocess
    pkgs = ["ecmwf-opendata", "cfgrib", "eccodes"]
    for pkg in pkgs:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True)
    print("  Realtime deps installed")


# ── Mode Logic ────────────────────────────────────────
if FORECAST_MODE == "historical":
    INIT_DATE = HISTORICAL_ANALOG_DATE
    print(f"""
  NOTE: Using 2020 historical analog for March 2026 forecast.
  Atmospheric patterns for late March are climatologically
  similar year-to-year. This gives a realistic forecast shaped
  but NOT identical to what will actually happen in 2026.
  For exact 2026 forecast use FORECAST_MODE = "realtime"
    """)
# realtime mode handled after ERA5 open below

MODEL_NAME = "v1/deterministic_2_8_deg.pkl"
ERA5_ZARR  = ("gs://gcp-public-data-arco-era5/ar/"
              "full_37-1h-0p25deg-chunk-1.zarr-v3")
LON_360    = LON % 360

DARK = "#0d1117"; PANEL = "#161b22"; BORDER = "#30363d"; W = "white"

# ════════════════════════════════════════════════════════════════
# 1. LOAD CHECKPOINT
# ════════════════════════════════════════════════════════════════
print(f"\n[1/7] Loading checkpoint: {MODEL_NAME}")
for attempt in range(3):
    try:
        gcs = gcsfs.GCSFileSystem(token="anon")
        with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
            ckpt = pickle.load(f)
        model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
        print(f"  OK | input_vars  = {model.input_variables}")
        print(f"     | forcing_vars = {model.forcing_variables}")
        break
    except Exception as e:
        print(f"  Attempt {attempt+1}: {e}"); time.sleep(5)
else:
    raise RuntimeError("Checkpoint load failed after 3 attempts")

# ════════════════════════════════════════════════════════════════
# 2. OPEN ERA5
# ════════════════════════════════════════════════════════════════
print(f"\n[2/7] Opening ERA5...")
for attempt in range(3):
    try:
        era5 = xr.open_zarr(ERA5_ZARR, chunks=None,
                             storage_options=dict(token="anon"))
        print(f"  OK | dims={dict(era5.sizes)}")
        print(f"     | vars={list(era5.data_vars)[:10]}...")
        break
    except Exception as e:
        print(f"  Attempt {attempt+1}: {e}"); time.sleep(5)
else:
    raise RuntimeError("ERA5 open failed")

# Detect ERA5 coordinate names
ERA5_LAT = "latitude" if "latitude" in era5.coords else "lat"
ERA5_LON = "longitude" if "longitude" in era5.coords else "lon"
print(f"  ERA5 coord names: lat='{ERA5_LAT}' lon='{ERA5_LON}'")

needed = list(set(model.input_variables) | set(model.forcing_variables))
print(f"  Needed vars: {needed}")

era5_grid = spherical_harmonic.Grid(
    latitude_nodes   = era5.sizes[ERA5_LAT],
    longitude_nodes  = era5.sizes[ERA5_LON],
    latitude_spacing = xarray_utils.infer_latitude_spacing(
        era5[ERA5_LAT]),
    longitude_offset = xarray_utils.infer_longitude_offset(
        era5[ERA5_LON]),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True)

# ════════════════════════════════════════════════════════════════
# 3. LOAD INIT STATE
# ════════════════════════════════════════════════════════════════

# Handle realtime mode ERA5 loading
if FORECAST_MODE == "realtime":
    install_realtime_deps()
    grib_file = download_ecmwf_init(TODAY)
    if grib_file:
        era5_slice = grib_to_era5_format(grib_file)
        if era5_slice is None:
            print("  Falling back to historical analog")
            FORECAST_MODE = "historical"
            INIT_DATE = HISTORICAL_ANALOG_DATE
    else:
        print("  Falling back to historical analog")
        FORECAST_MODE = "historical"
        INIT_DATE = HISTORICAL_ANALOG_DATE

init_dt = pd.Timestamp(INIT_DATE)

fc_end = init_dt + pd.Timedelta(days=FORECAST_DAYS)
mode_label = "Historical analog (ERA5)" if FORECAST_MODE == "historical" \
             else "Real-time (ECMWF open data)"
print(f"""
╔══════════════════════════════════════════════════════════╗
║  FORECAST CONFIGURATION                                 ║
║  Location  : {LOCATION_NAME:<42}║
║  Lat/Lon   : {LAT:.4f}°N, {LON:.4f}°E{'':<24}║
║  From      : {init_dt.strftime('%d %B %Y'):<42}║
║  To        : {fc_end.strftime('%d %B %Y'):<42}║
║  Days      : {FORECAST_DAYS} days ahead{'':<34}║
║  Mode      : {mode_label:<42}║
╚══════════════════════════════════════════════════════════╝
""")

print(f"\n[3/7] ERA5 initial state at {init_dt}...")

if FORECAST_MODE != "realtime" or era5_slice is None:
    era5_slice = (era5[needed]
                  .sel(time=init_dt, method="nearest")
                  .compute())
print(f"  Slice dims: {dict(era5_slice.sizes)}")

print(f"  Regridding ERA5 0.25deg to NeuralGCM 2.8deg...")
ev = xarray_utils.regrid(era5_slice, regridder)
ev = xarray_utils.fill_nan_with_nearest(ev)
print(f"  Regridded: {dict(ev.sizes)}")

# ════════════════════════════════════════════════════════════════
# 4. RUN FORECAST — CORRECT unroll() CALL
# ════════════════════════════════════════════════════════════════
print(f"\n[4/7] Running {FORECAST_DAYS}-day forecast...")
t0 = time.time()

# CORRECT: encode takes raw inputs/forcings without time axis
inputs   = model.inputs_from_xarray(ev)
forcings = model.forcings_from_xarray(ev)
state    = model.encode(inputs, forcings, jax.random.key(0))

# For unroll, TemporalForcings are required (with a time dimension).
# So we expand dimensions specifically for the unroll pass.
temporal_forcings = {k: jnp.expand_dims(jnp.asarray(v), 0) for k, v in forcings.items()}

_, preds = model.unroll(
    state,
    temporal_forcings,
    steps=FORECAST_DAYS + 1,
    timedelta=np.timedelta64(24, "h"),
    start_with_input=True,
)
print(f"  Unroll done in {time.time()-t0:.1f}s")
print(f"  preds type: {type(preds)}")
print(f"  preds keys: {list(preds.keys()) if hasattr(preds,'keys') else 'N/A'}")

td = pd.to_timedelta(np.arange(FORECAST_DAYS + 1) * 24, "h")

# CORRECT: pass preds directly
try:
    ds = model.data_to_xarray(preds, times=td)
except Exception as e:
    print(f"  data_to_xarray failed ({e}), trying without sim_time...")
    if hasattr(preds, "_asdict"):
        preds_dict = {k: v for k, v in preds._asdict().items()
                      if k != "sim_time"}
    elif hasattr(preds, "__dict__"):
        preds_dict = {k: v for k, v in preds.__dict__.items()
                      if k != "sim_time"}
    elif isinstance(preds, dict):
        preds_dict = {k: v for k, v in preds.items()
                      if k != "sim_time"}
    else:
        raise
    ds = model.data_to_xarray(preds_dict, times=td)

print(f"  Output vars: {list(ds.data_vars)}")
print(f"  Output dims: {dict(ds.sizes)}")

# CRITICAL: print actual dim names so we know what to use
sample = list(ds.data_vars.values())[0]
print(f"  Sample var '{list(ds.data_vars)[0]}' dims: {sample.dims}")
print(f"  Sample var coords: {list(sample.coords)}")

forecast_dates = [init_dt + pd.Timedelta(days=d)
                  for d in range(FORECAST_DAYS + 1)]

# ════════════════════════════════════════════════════════════════
# 5. DETECT DIMS AND BUILD EXTRACTOR
# ════════════════════════════════════════════════════════════════
LAT_DIM = LON_DIM = TIME_DIM = None
for da in ds.data_vars.values():
    for d in da.dims:
        dl = d.lower()
        if "lat" in dl and LAT_DIM is None:
            LAT_DIM = d
        if "lon" in dl and LON_DIM is None:
            LON_DIM = d
        if ("time" in dl or "delta" in dl or
                "step" in dl) and TIME_DIM is None:
            TIME_DIM = d
    if LAT_DIM and LON_DIM and TIME_DIM:
        break

# Fallback dim names
LAT_DIM  = LAT_DIM  or "latitude"
LON_DIM  = LON_DIM  or "longitude"
TIME_DIM = TIME_DIM or "prediction_timedelta"
print(f"\n  Using dims: time='{TIME_DIM}' "
      f"lat='{LAT_DIM}' lon='{LON_DIM}'")

# Build grid from actual coords
def build_grid():
    for da in ds.data_vars.values():
        if LAT_DIM in da.coords and LON_DIM in da.coords:
            lats = np.array(da.coords[LAT_DIM])
            lons = np.array(da.coords[LON_DIM])
            # ensure lons are 0-360
            lons = lons % 360
            return lats, lons
    # Fallback: Gaussian T63 grid
    lats = np.linspace(87.863, -87.863, 64)
    lons = np.linspace(0, 357.1875, 128)
    return lats, lons

GRID_LATS, GRID_LONS = build_grid()
LAT_IDX   = int(np.argmin(np.abs(GRID_LATS - LAT)))
LON_IDX   = int(np.argmin(np.abs(GRID_LONS - LON_360)))
MODEL_LAT = float(GRID_LATS[LAT_IDX])
MODEL_LON = float(GRID_LONS[LON_IDX])
print(f"  Requested  : {LAT:.4f}N {LON:.4f}E")
print(f"  Model grid : {MODEL_LAT:.2f}N {MODEL_LON:.2f}E "
      f"[idx {LAT_IDX},{LON_IDX}]")

N = FORECAST_DAYS + 1

def extract(var, level=None):
    """Extract location time series from NeuralGCM forecast."""
    if var not in ds.data_vars:
        return None
    da = ds[var]
    # Build isel dict using detected dim names
    idx = {TIME_DIM: slice(0, N)}
    if LAT_DIM in da.dims:
        idx[LAT_DIM] = LAT_IDX
    if LON_DIM in da.dims:
        idx[LON_DIM] = LON_IDX
    da = da.isel(**idx)
    if level is not None and "level" in da.dims:
        da = da.sel(level=level, method="nearest")
    arr = np.array(da).flatten()
    return arr[:N] if len(arr) >= N else np.pad(
        arr, (0, N - len(arr)), constant_values=np.nan)

def era5_pt(var, level=None):
    """Pull ERA5 point values at location for all forecast dates."""
    vals = []
    for dt in forecast_dates:
        try:
            da = era5[var].sel(time=dt, method="nearest")
            if level is not None and "level" in da.dims:
                da = da.sel(level=level, method="nearest")
            da = da.sel({ERA5_LAT: LAT, ERA5_LON: LON_360},
                        method="nearest")
            vals.append(float(da.compute()))
        except Exception as ex:
            vals.append(np.nan)
    return np.array(vals)

# ════════════════════════════════════════════════════════════════
# 6. EXTRACT ALL WEATHER VARIABLES
# ════════════════════════════════════════════════════════════════
print(f"\n[5/7] Extracting weather variables at {LOCATION_NAME}...")

# ── Temperature (K → °C) ──────────────────────────────────────
T_K_850 = extract("temperature", 850)
T_K_500 = extract("temperature", 500)
T_C_850 = T_K_850 - 273.15 if T_K_850 is not None else None
T_C_500 = T_K_500 - 273.15 if T_K_500 is not None else None
E5_T_850 = era5_pt("temperature", 850)
E5_T_C   = E5_T_850 - 273.15
print(f"  T850(C) forecast: {T_C_850}")
print(f"  T850(C) ERA5    : {E5_T_C}")

# ── Temperature comparison debug (BUG C) ──────────────────────
print(f"  Temperature comparison at 850 hPa:")
print(f"    NeuralGCM: {T_C_850}")
print(f"    ERA5 truth: {E5_T_C}")
print(f"    Gap (K)   : {T_C_850 - E5_T_C if T_C_850 is not None else 'N/A'}")
if T_C_850 is not None and not np.all(np.isnan(E5_T_C)):
    mean_gap = np.nanmean(np.abs(T_C_850 - E5_T_C))
    print(f"    Mean abs gap: {mean_gap:.2f} K  "
          f"(paper benchmark Day-1: ~0.8K)")

# ── Specific Humidity → Relative Humidity ─────────────────────
Q_850 = extract("specific_humidity", 850)
Q_500 = extract("specific_humidity", 500)
E5_Q  = era5_pt("specific_humidity", 850)
print(f"  Q_850 raw: {Q_850}")

def q_to_rh(q, T_K, p=850):
    if q is None or T_K is None:
        return None
    T_C = T_K - 273.15
    es  = 6.1078 * np.exp(17.27 * T_C / (T_C + 237.3))
    qs  = 0.622 * es / (p - es)
    return np.clip(q / qs * 100, 0, 100)

RH_850  = q_to_rh(Q_850, T_K_850, 850)
RH_500  = q_to_rh(Q_500, T_K_500, 500)
E5_RH   = q_to_rh(E5_Q, E5_T_850, 850)
print(f"  RH850(%): {RH_850}")

# ── Wind at 850, 500, 250 hPa ─────────────────────────────────
U_850 = extract("u_component_of_wind", 850)
V_850 = extract("v_component_of_wind", 850)
U_500 = extract("u_component_of_wind", 500)
V_500 = extract("v_component_of_wind", 500)
U_250 = extract("u_component_of_wind", 250)
V_250 = extract("v_component_of_wind", 250)

def ws(u, v):
    return np.sqrt(u**2 + v**2) if u is not None and v is not None else None

WS_850 = ws(U_850, V_850)
WS_500 = ws(U_500, V_500)
WS_250 = ws(U_250, V_250)
E5_U   = era5_pt("u_component_of_wind", 850)
E5_V   = era5_pt("v_component_of_wind", 850)
E5_WS  = np.sqrt(E5_U**2 + E5_V**2)
print(f"  WS850(m/s): {WS_850}")

# ── Wind Direction ─────────────────────────────────────────────
def wdir(u, v):
    if u is None or v is None:
        return None
    return (270 - np.rad2deg(np.arctan2(v, u))) % 360

def wdir_compass(deg):
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[int(round(float(deg) + 11.25) // 22) % 16]

WDIR_850 = wdir(U_850, V_850)
WDIR_500 = wdir(U_500, V_500)

# ── Geopotential Height ────────────────────────────────────────
Z500_m = Z850_m = None
Z_500  = extract("geopotential", 500)
Z_850  = extract("geopotential", 850)
if Z_500 is not None:
    Z500_m = Z_500 / 9.80665
if Z_850 is not None:
    Z850_m = Z_850 / 9.80665
E5_Z500   = era5_pt("geopotential", 500)
E5_Z500_m = E5_Z500 / 9.80665 if not np.all(np.isnan(E5_Z500)) else None
print(f"  Z500(m): {Z500_m}")

# ── Total Precipitable Water ───────────────────────────────────
def precipitable_water():
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
    # Collect q at each available level (avoid double extract calls)
    q_cols = []
    for lev in levels:
        q_lev = extract("specific_humidity", lev)
        if q_lev is not None:
            q_cols.append((lev, q_lev))
    if len(q_cols) < 2:
        return None
    # Trapezoidal integration: PW = (1/g) * integral(q * dp)
    pw = np.zeros(N)
    for i in range(len(q_cols) - 1):
        p1, q1 = q_cols[i]
        p2, q2 = q_cols[i + 1]
        dp = abs(p2 - p1) * 100   # hPa → Pa
        pw += (q1 + q2) / 2 * dp / 9.80665
    # PW is now in kg/m², which equals mm of water
    return pw

PW = precipitable_water()
if PW is not None:
    pw_mean = np.nanmean(PW)
    if pw_mean > 500:
        print(f"  WARNING: TPW={pw_mean:.0f}mm is physically "
              f"impossible. Bug: * 1000 still present.")
        PW = PW / 1000
        print(f"  Auto-corrected TPW to {np.nanmean(PW):.1f}mm")
    elif pw_mean < 1:
        print(f"  WARNING: TPW={pw_mean:.3f}mm is too low. "
              f"Check q units — may be in g/kg not kg/kg.")
    else:
        print(f"  TPW OK: {pw_mean:.1f}mm (expected 30-80mm tropics)")

# ── Surface Pressure (BUG B debug) ────────────────────────────
SP_hPa = None
# Try log_surface_pressure first, then surface_pressure, then mean_sea_level_pressure
LOG_PS = extract("log_surface_pressure")
if LOG_PS is not None:
    print(f"  LOG_PS raw values: {LOG_PS}")
    print(f"  LOG_PS mean: {np.nanmean(LOG_PS):.4f}")
    sp_candidate = np.exp(LOG_PS) / 100.0
    print(f"  SP candidate (hPa): {sp_candidate}")
    if 800 < np.nanmean(sp_candidate) < 1100:
        SP_hPa = sp_candidate
        print(f"  SP accepted: mean={np.nanmean(SP_hPa):.1f} hPa")
    else:
        # Try: maybe stored as Pa directly not log(Pa)
        sp_candidate2 = LOG_PS / 100.0
        if 800 < np.nanmean(sp_candidate2) < 1100:
            SP_hPa = sp_candidate2
            print(f"  SP fixed (direct Pa/100): {SP_hPa}")
        else:
            print(f"  SP rejected — trying other sources")
            SP_hPa = None
if SP_hPa is None:
    SP_direct = extract("surface_pressure")
    if SP_direct is not None:
        SP_raw = SP_direct / 100.0   # Pa → hPa
        if 800 < np.nanmean(SP_raw) < 1100:
            SP_hPa = SP_raw
        else:
            print(f"  WARNING: SP from surface_pressure sanity check failed: "
                  f"mean={np.nanmean(SP_raw):.1f} hPa — skipping")
if SP_hPa is None:
    SP_mslp = extract("mean_sea_level_pressure")
    if SP_mslp is not None:
        SP_raw = SP_mslp / 100.0
        if 800 < np.nanmean(SP_raw) < 1100:
            SP_hPa = SP_raw
if SP_hPa is None:
    print(f"  NOTE: Surface pressure not in model output vars: {list(ds.data_vars)}")
    print(f"        Will use ERA5 surface pressure for plotting reference only")
E5_SP = era5_pt("surface_pressure") / 100.0
print(f"  SP(hPa): {SP_hPa}")

# ── Atmospheric Stability (Lapse Rate) ─────────────────────────
def lapse_rate(T_low_K, T_high_K, p_low=850, p_high=500):
    if T_low_K is None or T_high_K is None:
        return None
    T_mean = (T_low_K + T_high_K) / 2
    dz = (287.05 * T_mean / 9.80665) * np.log(p_low / p_high)
    dT = T_low_K - T_high_K
    return (dT / dz) * 1000   # °C/km

LAPSE = lapse_rate(T_K_850, T_K_500)
print(f"  Lapse rate(°C/km): {LAPSE}")

# ── Cloud Water Content ────────────────────────────────────────
# NeuralGCM predicts specific cloud liquid and ice water content
CLWC_850 = extract("specific_cloud_liquid_water_content", 850)
CIWC_850 = extract("specific_cloud_ice_water_content", 850)
# Convert to g/kg for readability
CLWC_gkg = CLWC_850 * 1000 if CLWC_850 is not None else None
CIWC_gkg = CIWC_850 * 1000 if CIWC_850 is not None else None
print(f"  CLWC(g/kg): {CLWC_gkg}")
print(f"  CIWC(g/kg): {CIWC_gkg}")

# ── Vorticity and Divergence ──────────────────────────────────
VORT_850 = extract("vorticity", 850)
DIV_850  = extract("divergence", 850)
print(f"  Vorticity: {VORT_850}")

# ════════════════════════════════════════════════════════════════
# 7. DASHBOARD — 11 individual images
# ════════════════════════════════════════════════════════════════
print(f"\n[6/7] Building dashboard (11 individual images)...")

X = np.arange(N)
date_labels = [dt.strftime("%a\n%d %b") for dt in forecast_dates]

# ── Output folder ─────────────────────────────────────────────
safe    = LOCATION_NAME.replace(",","").replace(" ","_")
out_dir = f"forecast_{safe}_{init_dt.strftime('%Y%m%d')}"
os.makedirs(out_dir, exist_ok=True)

# ── Shared helpers ────────────────────────────────────────────
def new_fig(title_short):
    """Create a single-panel figure with dark theme + title bar."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK)
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.95)
    fig.text(0.5, 0.98, "NeuralGCM  5-Day Weather Forecast",
             ha="center", va="top", color=W,
             fontsize=16, fontweight="bold")
    fig.text(0.5, 0.945,
             f"{LOCATION_NAME}   {LAT:.4f}°N, {LON:.4f}°E  |  "
             f"{init_dt.strftime('%d %b %Y')} → "
             f"{fc_end.strftime('%d %b %Y')}  |  {mode_label}",
             ha="center", va="top", color="#8B949E", fontsize=9)
    return fig, ax

def S(ax, title, ylab, unit, note=""):
    ax.set_facecolor(PANEL)
    t = f"{title}\n📍 {LOCATION_NAME}"
    if note: t += f"\n({note})"
    ax.set_title(t, color="#58A6FF", fontsize=11,
                 fontweight="bold", pad=8, loc="left")
    ax.set_ylabel(f"{ylab} [{unit}]", color=W, fontsize=10)
    ax.set_xticks(X); ax.set_xticklabels(date_labels,
                                          color=W, fontsize=9)
    ax.tick_params(axis="y", colors=W, labelsize=9)
    ax.grid(True, color=BORDER, ls="--", alpha=0.5, lw=0.6)
    [sp.set_edgecolor(BORDER) for sp in ax.spines.values()]
    ax.set_xlim(-0.4, FORECAST_DAYS + 0.4)

def badge(ax):
    ax.text(0.99, 0.97, "NeuralGCM", transform=ax.transAxes,
            color="#3FB950", fontsize=8, ha="right", va="top",
            bbox=dict(fc=PANEL, ec="#3FB950",
                      boxstyle="round,pad=0.2"))

def add_mae_badge(ax, fc_vals, era5_vals, unit):
    if fc_vals is None or era5_vals is None:
        return
    valid = ~(np.isnan(fc_vals) | np.isnan(era5_vals))
    if valid.sum() == 0:
        return
    mae = np.mean(np.abs(fc_vals[valid] - era5_vals[valid]))
    ax.text(0.99, 0.03,
            f"MAE vs ERA5: {mae:.2f} {unit}",
            transform=ax.transAxes,
            color="#EF9F27", fontsize=8,
            ha="right", va="top",
            bbox=dict(fc=PANEL, ec="#EF9F27",
                      boxstyle="round,pad=0.2"))

def ann(ax, arr, c, fmt=".1f", dy=8):
    if arr is None: return
    for xi, v in zip(X, arr):
        if not np.isnan(v):
            ax.annotate(f"{v:{fmt}}", xy=(xi,v),
                        xytext=(0,dy),
                        textcoords="offset points",
                        color=c, fontsize=8,
                        fontweight="bold", ha="center")

def save_panel(fig, num, name):
    path = os.path.join(out_dir, f"{num:02d}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════
# PANEL 1: Temperature
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Temperature")
if T_C_850 is not None:
    ax.plot(X, T_C_850, color="#F78166", lw=2.5,
            marker="o", ms=7, label="NeuralGCM 850 hPa (~1500m)",
            zorder=3)
    ax.fill_between(X, T_C_850-2, T_C_850+2,
                    color="#F78166", alpha=0.12,
                    label="±2°C uncertainty band")
    ann(ax, T_C_850, "#F78166")
if T_C_500 is not None:
    ax.plot(X, T_C_500, color="#EF9F27", lw=1.8,
            marker="s", ms=5, ls="--",
            label="NeuralGCM 500 hPa (mid-atm)", alpha=0.85)
ax.plot(X, E5_T_C, color=W, lw=1.5, marker="^",
        ms=5, ls=":", label="ERA5 truth 850 hPa", alpha=0.7)
S(ax, "Temperature",
  "Temperature", "°C",
  "850 hPa ≈ near-surface proxy  |  500 hPa = mid-troposphere\n"
  "ERA5 dashed = observed truth for model verification")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER, loc="best")
badge(ax)
add_mae_badge(ax, T_C_850, E5_T_C, "°C")
save_panel(fig, 1, "temperature")

# ══════════════════════════════════════════════════════════════
# PANEL 2: Geopotential Height Z500
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Geopotential Height")
if Z500_m is not None:
    ax.plot(X, Z500_m, color="#BD8EE6", lw=2.5,
            marker="o", ms=7, label="NeuralGCM Z500", zorder=3)
    ann(ax, Z500_m, "#BD8EE6", fmt=".0f")
if E5_Z500_m is not None:
    ax.plot(X, E5_Z500_m, color=W, lw=1.5, marker="^",
            ms=5, ls=":", label="ERA5 truth", alpha=0.7)
ax.axhline(5500, color="#BD8EE6", lw=0.8, ls=":", alpha=0.4,
           label="5500m = tropical threshold")
S(ax, "Geopotential Height at 500 hPa",
  "Z500 Height", "m",
  "PRIMARY synoptic indicator  |  LOW=troughs/rain  HIGH=ridges/clear\n"
  "Tropics: ~5700-5900m  |  Mid-lat lows: <5400m  |  Blocking: >5900m")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER)
badge(ax)
add_mae_badge(ax, Z500_m, E5_Z500_m, "m")
save_panel(fig, 2, "geopotential_height")

# ══════════════════════════════════════════════════════════════
# PANEL 3: Relative Humidity
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Relative Humidity")
if RH_850 is not None:
    ax.bar(X-0.15, RH_850, 0.28, color="#58A6FF",
           alpha=0.8, label="RH 850 hPa (low-level moisture)")
    ax.plot(X-0.15, RH_850, color="#58A6FF", lw=2,
            marker="o", ms=5, zorder=3)
if RH_500 is not None:
    ax.bar(X+0.15, RH_500, 0.28, color="#1D9E75",
           alpha=0.75, label="RH 500 hPa (mid-level moisture)")
if E5_RH is not None:
    ax.plot(X, E5_RH, color=W, lw=1.5, marker="^",
            ms=5, ls=":", label="ERA5 truth 850hPa", alpha=0.7)
ax.axhline(80, color="#F78166", lw=0.9, ls="--",
           alpha=0.6, label="80% = rain likely")
ax.axhline(40, color="#EF9F27", lw=0.9, ls="--",
           alpha=0.6, label="40% = semi-arid")
ax.set_ylim(0, 115)
S(ax, "Relative Humidity",
  "Relative Humidity", "%",
  "850 hPa = low-level clouds/fog  |  500 hPa = mid-level clouds\n"
  ">80%=expect rain  60-80%=cloudy  <40%=dry/clear")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER)
badge(ax)
save_panel(fig, 3, "relative_humidity")

# ══════════════════════════════════════════════════════════════
# PANEL 4: Total Precipitable Water
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Precipitable Water")
if PW is not None:
    ax.fill_between(X, 0, PW, color="#A5D6FF", alpha=0.4)
    ax.plot(X, PW, color="#A5D6FF", lw=2.5,
            marker="o", ms=7, label="NeuralGCM TPW", zorder=3)
    ann(ax, PW, "#A5D6FF", fmt=".1f")
ax.axhline(60, color="#F78166", lw=0.9, ls="--",
           alpha=0.6, label=">60mm = heavy rain risk")
ax.axhline(30, color="#EF9F27", lw=0.9, ls="--",
           alpha=0.6, label="<30mm = dry conditions")
ax.axhline(20, color="#58A6FF", lw=0.9, ls="--",
           alpha=0.6, label="<20mm = very dry / desert")
ax.set_ylim(bottom=0)
S(ax, "Total Precipitable Water (TPW)",
  "Water vapour column", "mm",
  "Vertical integral of specific humidity through entire atmosphere\n"
  "Direct proxy for potential rainfall amount and thunderstorm risk")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER)
badge(ax)
save_panel(fig, 4, "precipitable_water")

# ══════════════════════════════════════════════════════════════
# PANEL 5: Wind Speed
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Wind Speed")
if WS_850 is not None:
    ax.plot(X, WS_850, color="#3FB950", lw=2.5,
            marker="o", ms=6, label="850 hPa (~1500m surface winds)")
    ann(ax, WS_850, "#3FB950")
if WS_500 is not None:
    ax.plot(X, WS_500, color="#EF9F27", lw=2,
            marker="s", ms=5, ls="--",
            label="500 hPa (steering-level winds)")
if WS_250 is not None:
    ax.plot(X, WS_250, color="#F78166", lw=2,
            marker="D", ms=5, ls="-.",
            label="250 hPa (jet stream level)")
ax.plot(X, E5_WS, color=W, lw=1.5, marker="^",
        ms=5, ls=":", label="ERA5 850hPa truth", alpha=0.7)
ax.axhline(17, color="#EF9F27", lw=0.8, ls=":", alpha=0.5,
           label="17 m/s = Beaufort 8 (gale)")
ax.axhline(33, color="#F78166", lw=0.8, ls=":", alpha=0.5,
           label="33 m/s = hurricane force")
ax.set_ylim(bottom=0)
S(ax, "Wind Speed at 3 Atmospheric Levels",
  "Wind Speed", "m/s",
  "850hPa=near-surface winds felt at ground  "
  "500hPa=steers weather systems\n"
  "250hPa=jet stream (determines storm track speed)")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER, loc="best")
badge(ax)
add_mae_badge(ax, WS_850, E5_WS, "m/s")
save_panel(fig, 5, "wind_speed")

# ══════════════════════════════════════════════════════════════
# PANEL 6: Wind Direction
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Wind Direction")
ax.set_facecolor(PANEL)
if WDIR_850 is not None and WS_850 is not None:
    sc = ax.scatter(X, WDIR_850, c=WS_850,
                    cmap="RdYlGn_r", s=180,
                    vmin=0, vmax=25, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label("Wind speed (m/s)", color=W, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=W)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=W)
    for xi, (d, ws_v) in enumerate(zip(WDIR_850, WS_850)):
        comp = wdir_compass(d)
        ax.annotate(f"{comp}\n{d:.0f}°",
                    xy=(xi, d), xytext=(0,12),
                    textcoords="offset points",
                    color=W, fontsize=8, ha="center")
for deg, lbl in [(0,"N"),(90,"E"),(180,"S"),(270,"W"),(360,"N")]:
    ax.axhline(deg, color=BORDER, lw=0.5, alpha=0.5)
    ax.text(FORECAST_DAYS+0.35, deg, lbl,
            color="#8B949E", fontsize=9, va="center")
ax.set_ylim(-15, 390)
S(ax, "Wind Direction at 850 hPa",
  "Wind Direction", "°",
  "0°/360°=N  90°=E  180°=S  270°=W\n"
  "Colour = wind speed (green=calm → red=strong)")
ax.set_yticks([0, 90, 180, 270, 360])
ax.set_yticklabels(["0° N", "90° E", "180° S",
                    "270° W", "360° N"],
                   color=W, fontsize=9)
badge(ax)
save_panel(fig, 6, "wind_direction")

# ══════════════════════════════════════════════════════════════
# PANEL 7: Wind Components U/V
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Wind Components")
if U_850 is not None:
    ax.plot(X, U_850, color="#EF9F27", lw=2.5,
            marker="o", ms=6,
            label="U 850hPa (+ = westerly / blowing east)")
if V_850 is not None:
    ax.plot(X, V_850, color="#D85A30", lw=2.5,
            marker="s", ms=6,
            label="V 850hPa (+ = southerly / blowing north)")
if U_500 is not None:
    ax.plot(X, U_500, color="#EF9F27", lw=1.5,
            marker="o", ms=4, ls="--",
            label="U 500hPa", alpha=0.6)
if V_500 is not None:
    ax.plot(X, V_500, color="#D85A30", lw=1.5,
            marker="s", ms=4, ls="--",
            label="V 500hPa", alpha=0.6)
ax.axhline(0, color=W, lw=0.8, alpha=0.3)
S(ax, "Wind Components U (East-West) and V (North-South)",
  "Wind component", "m/s",
  "+U=westerly wind  -U=easterly  +V=southerly  -V=northerly\n"
  "Monsoon: strong +V at 850hPa (warm moist onshore flow)")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER, loc="best")
badge(ax)
save_panel(fig, 7, "wind_components")

# ══════════════════════════════════════════════════════════════
# PANEL 8: Cloud Water Content
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Cloud Water")
if CLWC_gkg is not None:
    ax.fill_between(X, 0, CLWC_gkg, color="#A5D6FF",
                    alpha=0.5, label="Cloud liquid water 850hPa")
    ax.plot(X, CLWC_gkg, color="#A5D6FF", lw=2,
            marker="o", ms=5, zorder=3)
    ann(ax, CLWC_gkg, "#A5D6FF", fmt=".4f", dy=6)
if CIWC_gkg is not None:
    ax.fill_between(X, 0, CIWC_gkg, color="#BD8EE6",
                    alpha=0.4, label="Cloud ice water 850hPa")
    ax.plot(X, CIWC_gkg, color="#BD8EE6", lw=2,
            marker="s", ms=5, zorder=3)
if CLWC_gkg is None and CIWC_gkg is None:
    ax.text(0.5, 0.5,
            "Cloud water not in\nmodel output vars",
            transform=ax.transAxes,
            color="#8B949E", ha="center", va="center",
            fontsize=11)
ax.set_ylim(bottom=0)
S(ax, "Cloud Water Content at 850 hPa",
  "Cloud water", "g/kg",
  "NeuralGCM predicts liquid and ice cloud water content\n"
  "Higher values = thicker clouds = more rain potential")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER)
badge(ax)
save_panel(fig, 8, "cloud_water")

# ══════════════════════════════════════════════════════════════
# PANEL 9: Atmospheric Stability (Lapse Rate)
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Atmospheric Stability")
if LAPSE is not None:
    lapse_colors = ["#F78166" if l > 9.8
                    else "#EF9F27" if l > 7.0
                    else "#3FB950" for l in LAPSE]
    ax.bar(X, LAPSE, 0.5, color=lapse_colors, alpha=0.85)
    ax.plot(X, LAPSE, color=W, lw=1.5, marker="o", ms=5, zorder=3)
    ann(ax, LAPSE, W, fmt=".2f")
ax.axhline(9.8, color="#F78166", lw=1.2, ls="--",
           label="9.8°C/km DALR (absolutely unstable)")
ax.axhline(6.5, color="#3FB950", lw=1.2, ls="--",
           label="6.5°C/km standard (neutral)")
ax.axhline(5.0, color="#58A6FF", lw=1.2, ls="--",
           label="5.0°C/km SALR (stable)")
S(ax, "Atmospheric Stability (Lapse Rate 850→500 hPa)",
  "Lapse rate", "°C/km",
  "RED >9.8 = absolutely unstable = thunderstorm risk\n"
  "ORANGE 7-9.8 = conditionally unstable  GREEN <6.5 = stable/clear")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER)
badge(ax)
save_panel(fig, 9, "atmospheric_stability")

# ══════════════════════════════════════════════════════════════
# PANEL 10: Surface Pressure
# ══════════════════════════════════════════════════════════════
fig, ax = new_fig("Surface Pressure")
if SP_hPa is not None:
    ax.plot(X, SP_hPa, color="#BD8EE6", lw=2.5,
            marker="o", ms=7, label="NeuralGCM MSLP", zorder=3)
    ax.fill_between(X, SP_hPa-2, SP_hPa+2,
                    color="#BD8EE6", alpha=0.12)
    ann(ax, SP_hPa, "#BD8EE6", fmt=".1f")
ax.plot(X, E5_SP, color=W, lw=1.5, marker="^",
        ms=5, ls=":", label="ERA5 truth", alpha=0.75)
ax.axhline(1013.25, color=W, lw=0.8, ls=":", alpha=0.4,
           label="1013.25 hPa standard atmosphere")
ax.axhline(1000, color="#F78166", lw=0.8, ls="--", alpha=0.4,
           label="<1000 hPa = low pressure system")
ax.axhline(1020, color="#3FB950", lw=0.8, ls="--", alpha=0.4,
           label=">1020 hPa = high pressure / fair weather")
S(ax, "Mean Sea-Level Pressure (MSLP)",
  "Pressure", "hPa",
  "Falling pressure = storm/cyclone approaching\n"
  "Rising pressure = clearing weather ahead")
ax.legend(fontsize=8, facecolor=PANEL, labelcolor=W,
          edgecolor=BORDER)
badge(ax)
add_mae_badge(ax, SP_hPa, E5_SP, "hPa")
save_panel(fig, 10, "surface_pressure")

# ══════════════════════════════════════════════════════════════
# PANEL 11: Forecast Summary Table
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK)
fig.subplots_adjust(top=0.82, bottom=0.05, left=0.04, right=0.96)
fig.text(0.5, 0.97, "NeuralGCM  Forecast Summary",
         ha="center", va="top", color=W,
         fontsize=16, fontweight="bold")
fig.text(0.5, 0.92,
         f"{LOCATION_NAME}  ({LAT:.2f}°N, {LON:.2f}°E)  |  "
         f"{init_dt.strftime('%d %b %Y')} → "
         f"{fc_end.strftime('%d %b %Y')}  |  {mode_label}",
         ha="center", va="top", color="#8B949E", fontsize=10)
ax.set_facecolor(PANEL); ax.axis("off")

col_h = ["Date","T850 °C","RH850 %","WS850 m/s",
         "Dir","TPW mm","Z500 m","SP hPa","Stability"]
rows = []
for d, dt in enumerate(forecast_dates):
    stab = "—"
    if LAPSE is not None and not np.isnan(LAPSE[d]):
        if LAPSE[d] > 9.8:   stab = "UNSTABLE"
        elif LAPSE[d] > 7.0: stab = "Cond.unst"
        else:                 stab = "Stable"
    rows.append([
        dt.strftime("%a %d %b"),
        f"{T_C_850[d]:.1f}"   if T_C_850  is not None else "—",
        f"{RH_850[d]:.0f}"    if RH_850   is not None else "—",
        f"{WS_850[d]:.1f}"    if WS_850   is not None else "—",
        wdir_compass(WDIR_850[d]) if WDIR_850 is not None else "—",
        f"{PW[d]:.1f}"        if PW       is not None else "—",
        f"{Z500_m[d]:.0f}"    if Z500_m   is not None else "—",
        f"{SP_hPa[d]:.1f}"    if SP_hPa  is not None else "—",
        stab,
    ])

tbl = ax.table(cellText=rows, colLabels=col_h,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
tbl.scale(1.0, 2.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(BORDER)
    if r == 0:
        cell.set_facecolor("#21262d")
        cell.set_text_props(color="#58A6FF",
                            fontweight="bold", fontsize=11)
    else:
        cell.set_facecolor(PANEL if r%2==0 else "#1c2128")
        cell.set_text_props(color=W, fontsize=11)

save_panel(fig, 11, "forecast_summary")
print(f"\n  All 11 images saved to: {out_dir}/")

# ════════════════════════════════════════════════════════════════
# 8. TERMINAL FORECAST TABLE
# ════════════════════════════════════════════════════════════════
print(f"\n[7/7] Forecast summary:")
print("=" * 85)
print(f"  {LOCATION_NAME}  |  {LAT}N {LON}E  |  "
      f"Date: {init_dt.strftime('%d %b %Y')} → {fc_end.strftime('%d %b %Y')}")
print("=" * 85)
print(f"  {'Date':<13} {'T°C':>7} {'RH%':>6} {'WS m/s':>8} "
      f"{'Dir':>5} {'TPW mm':>8} {'Z500 m':>8} "
      f"{'SP hPa':>8} {'Stability':<12}")
print("  " + "-" * 83)
for d, dt in enumerate(forecast_dates):
    stab = "—"
    if LAPSE is not None and not np.isnan(LAPSE[d]):
        stab = ("UNSTABLE" if LAPSE[d]>9.8
                else "Cond.unstable" if LAPSE[d]>7.0
                else "Stable")
    print(f"  {dt.strftime('%a %d %b'):<13}"
          f" {(f'{T_C_850[d]:.1f}' if T_C_850 is not None else 'N/A'):>7}"
          f" {(f'{RH_850[d]:.0f}' if RH_850 is not None else 'N/A'):>6}"
          f" {(f'{WS_850[d]:.1f}' if WS_850 is not None else 'N/A'):>8}"
          f" {(wdir_compass(WDIR_850[d]) if WDIR_850 is not None else 'N/A'):>5}"
          f" {(f'{PW[d]:.1f}' if PW is not None else 'N/A'):>8}"
          f" {(f'{Z500_m[d]:.0f}' if Z500_m is not None else 'N/A'):>8}"
          f" {(f'{SP_hPa[d]:.1f}' if SP_hPa is not None else 'N/A'):>8}"
          f"  {stab:<12}")
print("=" * 85)
print(f"""
  TO FORECAST A DIFFERENT LOCATION:
  Change lines 35-39 at the top of this file:
    LOCATION_NAME = "Tokyo, Japan"
    LAT           =  35.6762
    LON           =  139.6503
    INIT_DATE     = "2020-09-01T00:00"
    FORECAST_DAYS = 5

  Output: {out_dir}/
  Paper : Kochkov et al. 2024 arXiv:2311.07222
  Model : gs://neuralgcm/models/{MODEL_NAME}
  ERA5  : gs://gcp-public-data-arco-era5
""")