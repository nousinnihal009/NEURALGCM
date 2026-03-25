"""
NeuralGCM Weather — CLI Entry Point
=====================================
Usage:
  python run_forecast.py --lat 13.08 --lon 80.27 --name "Chennai"
  python run_forecast.py --lat 28.61 --lon 77.21 --name "Delhi" --days 7
  python run_forecast.py --schedule          (runs every 6h forever)
  python run_forecast.py --test              (runs unit tests)
  python run_forecast.py --date 2020-06-01   (historical mode)
"""

import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_FLAGS"]                     = "--xla_cpu_use_thunk_runtime=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import sys
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | "
                  "<level>{level:<8}</level> | {message}")

# Create logs dir if needed
os.makedirs("./logs", exist_ok=True)
logger.add("./logs/neuralgcm.log",
           level="DEBUG", rotation="10 MB", retention="7 days")


def main():
    parser = argparse.ArgumentParser(
        description="NeuralGCM Universal Weather Forecast")
    parser.add_argument("--lat",      type=float, default=13.0827)
    parser.add_argument("--lon",      type=float, default=80.2707)
    parser.add_argument("--name",     type=str,   default="Chennai, India")
    parser.add_argument("--days",     type=int,   default=None)
    parser.add_argument("--date",     type=str,   default=None,
                        help="Historical init date YYYY-MM-DD")
    parser.add_argument("--mode",     type=str,   default=None,
                        choices=["realtime", "historical"])
    parser.add_argument("--schedule", action="store_true",
                        help="Run automated scheduler every 6h")
    parser.add_argument("--test",     action="store_true",
                        help="Run unit tests")
    parser.add_argument("--no-save",  action="store_true",
                        help="Skip saving output files")
    args = parser.parse_args()

    # -- Run unit tests --
    if args.test:
        import pytest
        exit_code = pytest.main([
            "neuralgcm_weather/tests/",
            "-v", "--tb=short",
        ])
        sys.exit(exit_code)

    # -- Run scheduler --
    if args.schedule:
        from neuralgcm_weather.pipeline.scheduler import start_scheduler
        logger.info("Starting automated forecast scheduler...")
        start_scheduler(blocking=True)
        return

    # -- Run single forecast --
    from neuralgcm_weather.pipeline.forecast import run_forecast_pipeline

    mode = args.mode
    init_date = None
    if args.date:
        mode = "historical"
        init_date = f"{args.date}T00:00"
    elif mode is None:
        mode = "realtime"

    logger.info(
        f"Forecast: {args.name} | "
        f"({args.lat:.4f}N, {args.lon:.4f}E) | "
        f"mode={mode}")

    result = run_forecast_pipeline(
        location_name = args.name,
        lat           = args.lat,
        lon           = args.lon,
        forecast_days = args.days,
        init_date     = init_date,
        mode          = mode,
        save          = not args.no_save,
    )

    fp = result["forecast_point"]

    # Print clean summary table
    print(f"\n{'='*70}")
    print(f"  NeuralGCM Forecast — {fp.location_name}")
    print(f"  Init: {result['init_time'].strftime('%d %B %Y %H:%M UTC')}")
    print(f"  Mode: {result['mode_used']}")
    print(f"{'='*70}")
    print(f"  {'Date':<13} {'T°C':>7} {'RH%':>6} {'WS m/s':>8} "
          f"{'TPW mm':>8} {'Z500 m':>8} {'SP hPa':>8}")
    print(f"  {'-'*68}")

    import numpy as np
    for i, dt in enumerate(fp.dates):
        T  = (f"{fp.temperature_c_850[i]:.1f}"
              if fp.temperature_c_850 is not None else "N/A")
        RH = (f"{fp.rh_850[i]:.0f}"
              if fp.rh_850 is not None else "N/A")
        WS = (f"{fp.wind_speed_850[i]:.1f}"
              if fp.wind_speed_850 is not None else "N/A")
        PW = (f"{fp.tpw_mm[i]:.1f}"
              if fp.tpw_mm is not None else "N/A")
        Z5 = (f"{fp.z500_m[i]:.0f}"
              if fp.z500_m is not None else "N/A")
        SP = (f"{fp.mslp_hpa[i]:.1f}"
              if fp.mslp_hpa is not None else "N/A")
        print(f"  {dt.strftime('%a %d %b'):<13} "
              f"{T:>7} {RH:>6} {WS:>8} {PW:>8} {Z5:>8} {SP:>8}")

    print(f"{'='*70}")

    if result["violations"]:
        print(f"\n  WARNINGS ({len(result['violations'])}):")
        for v in result["violations"]:
            print(f"    {v}")

    if result["saved_files"]:
        print(f"\n  Saved:")
        for ftype, path in result["saved_files"].items():
            print(f"    {ftype}: {path}")


if __name__ == "__main__":
    main()
