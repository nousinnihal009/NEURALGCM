"""
Forecast Pipeline Orchestrator
================================
Coordinates: data loading -> regridding -> model inference ->
             extraction -> validation -> saving.
Single function run_forecast_pipeline() is the main entry point.
"""

import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_FLAGS"]                     = "--xla_cpu_use_thunk_runtime=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

from neuralgcm_weather.config import MODEL, DATA, OUTPUT, CFG_RAW
from neuralgcm_weather.model.checkpoint import load_checkpoint
from neuralgcm_weather.model.runner import (build_regridder,
                                            regrid_init_state,
                                            run_forecast)
from neuralgcm_weather.model.extractor import VariableExtractor
from neuralgcm_weather.output.validator import validate_forecast
from neuralgcm_weather.output.writer    import save_forecast


_ERA5     = None   # module-level singleton (lazy load)
_REGRIDDER = None  # built once after first ERA5 load


def get_era5():
    global _ERA5
    if _ERA5 is None:
        from neuralgcm_weather.data.era5_loader import open_era5
        _ERA5 = open_era5()
    return _ERA5


def run_forecast_pipeline(
    location_name: str,
    lat: float,
    lon: float,
    forecast_days: Optional[int]  = None,
    init_date:     Optional[str]  = None,
    mode:          Optional[str]  = None,
    save:          bool           = True,
) -> dict:
    """
    Run the complete NeuralGCM forecast pipeline.

    Args:
        location_name: human-readable name e.g. "Chennai, India"
        lat, lon:      coordinates
        forecast_days: override config default
        init_date:     "YYYY-MM-DDTHH:MM" for historical mode
        mode:          "realtime" or "historical" (overrides config)
        save:          whether to write output files

    Returns:
        dict with keys: forecast_point, saved_files,
                        elapsed_seconds, violations
    """
    global _REGRIDDER

    days = forecast_days or MODEL.forecast_days
    mode = mode or DATA.mode

    logger.info(
        f"Pipeline start | loc={location_name} | "
        f"mode={mode} | days={days}")

    # -- 1. Load model --
    model = load_checkpoint(
        model_name  = MODEL.checkpoint,
        gcs_bucket  = MODEL.gcs_bucket,
        local_cache_dir = DATA.cache_dir + "/checkpoints",
    )

    # -- 2. Load init state --
    init_date_str = None
    if mode == "realtime":
        from neuralgcm_weather.data.ecmwf_loader import (
            load_realtime_init_state)
        result = load_realtime_init_state(
            cache_dir   = DATA.cache_dir,
            lag_hours   = DATA.ecmwf_lag_hours,
        )
        if result is None:
            logger.warning(
                "ECMWF realtime failed. "
                "Falling back to historical analog.")
            mode = "historical"
        else:
            init_ds, init_time = result
            init_date_str = init_time.strftime("%Y-%m-%dT%H:%M")

    if mode == "historical":
        from neuralgcm_weather.data.era5_loader import (
            open_era5, load_era5_slice, get_seasonal_analog)
        era5 = get_era5()

        if init_date is None:
            # Use seasonal analog of today
            today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
            init_date_str = get_seasonal_analog(today, year=2020)
        else:
            init_date_str = init_date

        needed = list(set(model.input_variables) |
                      set(model.forcing_variables))
        init_ds = load_era5_slice(era5, init_date_str, needed)
        init_time = pd.Timestamp(init_date_str)

    # -- 3. Regrid --
    if _REGRIDDER is None:
        _REGRIDDER = build_regridder(init_ds, model)
    init_state = regrid_init_state(init_ds, _REGRIDDER)

    # -- 4. Run forecast --
    ds, elapsed = run_forecast(
        model,
        init_state,
        forecast_days  = days,
        timestep_hours = MODEL.timestep_hours,
    )

    # -- 5. Extract variables --
    forecast_dates = [
        init_time + pd.Timedelta(days=d)
        for d in range(days + 1)
    ]
    extractor = VariableExtractor(ds, days)
    fp = extractor.extract_all(
        location_name, lat, lon, forecast_dates)

    # -- 6. Validate --
    all_ok, violations = validate_forecast(fp)
    if violations:
        logger.warning(
            f"{len(violations)} sanity violations detected")

    # -- 7. Save --
    saved = {}
    if save:
        saved = save_forecast(
            fp,
            save_dir  = OUTPUT.save_dir,
            save_png  = OUTPUT.save_png,
            save_json = OUTPUT.save_json,
            save_csv  = OUTPUT.save_csv,
        )

    logger.success(
        f"Pipeline complete | {location_name} | "
        f"elapsed={elapsed:.1f}s | "
        f"sanity={'OK' if all_ok else 'WARN'}")

    return {
        "forecast_point":  fp,
        "saved_files":     saved,
        "elapsed_seconds": elapsed,
        "violations":      violations,
        "init_time":       init_time,
        "mode_used":       mode,
    }
