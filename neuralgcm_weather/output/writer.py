"""
Forecast Output Writer
======================
Saves forecast results to JSON, CSV, and PNG.
Organised by location and date.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime


def save_forecast(fp, save_dir: str = "./forecasts",
                  save_png: bool = True,
                  save_json: bool = True,
                  save_csv: bool = True) -> dict:
    """
    Save a ForecastPoint to disk.
    Returns dict of saved file paths.
    """
    safe_name = fp.location_name.replace(",","").replace(" ","_")
    init_str  = fp.dates[0].strftime("%Y%m%d")
    run_dir   = Path(save_dir) / safe_name / init_str
    run_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # -- JSON --
    if save_json:
        json_path = run_dir / "forecast.json"
        data = fp.to_dict()
        data["generated_at"] = datetime.utcnow().isoformat() + "Z"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        saved["json"] = str(json_path)
        logger.info(f"Saved JSON: {json_path}")

    # -- CSV --
    if save_csv:
        csv_path = run_dir / "forecast.csv"
        rows = []
        for i, dt in enumerate(fp.dates):
            row = {"date": dt.strftime("%Y-%m-%d")}
            for var in [
                "temperature_c_850","temperature_c_500",
                "rh_850","rh_500","tpw_mm",
                "wind_speed_850","wind_speed_500","wind_speed_250",
                "wind_dir_850","z500_m","mslp_hpa","lapse_rate",
            ]:
                arr = getattr(fp, var, None)
                row[var] = (round(float(arr[i]), 3)
                            if arr is not None and not np.isnan(arr[i])
                            else None)
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        saved["csv"] = str(csv_path)
        logger.info(f"Saved CSV: {csv_path}")

    # -- PNG --
    if save_png:
        try:
            from neuralgcm_weather.output.plotter import plot_forecast
            png_path = run_dir / "forecast.png"
            plot_forecast(fp, str(png_path))
            saved["png"] = str(png_path)
        except Exception as e:
            logger.warning(f"PNG generation failed: {e}")

    logger.success(
        f"Forecast saved for {fp.location_name} | "
        f"init={init_str} | files={list(saved.keys())}")
    return saved
