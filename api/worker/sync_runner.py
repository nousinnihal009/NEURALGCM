"""
Synchronous Forecast Runner
============================
Runs the NeuralGCM pipeline directly in-process (no Celery).
Used by the FastAPI router on Windows where Celery's prefork
pool is broken and module path isolation causes import failures.
"""

import sys
import os

# Ensure project root is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from datetime import datetime
from loguru import logger
import psycopg2
import json as json_mod


def _get_db_conn():
    """Create a synchronous psycopg2 connection."""
    from api.settings import get_settings
    s = get_settings()
    return psycopg2.connect(
        host=s.postgres_host, port=s.postgres_port,
        dbname=s.postgres_db, user=s.postgres_user,
        password=s.postgres_password)


def update_job_status(job_id: str, status: str,
                      result: dict = None, error: str = None):
    """Synchronous DB update for job status."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        if status == "running":
            cur.execute(
                "UPDATE forecast_runs SET status=%s, started_at=%s WHERE id=%s",
                (status, datetime.utcnow(), job_id))
        elif status == "complete":
            cur.execute(
                """UPDATE forecast_runs SET
                   status=%s, completed_at=%s, elapsed_sec=%s,
                   result=%s, sanity_ok=%s, sanity_violations=%s,
                   json_path=%s, csv_path=%s, png_path=%s,
                   model_lat=%s, model_lon=%s, init_time_utc=%s,
                   mode_used=%s, geom=ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                   WHERE id=%s""",
                (status, datetime.utcnow(),
                 result.get("elapsed_seconds"),
                 json_mod.dumps(result),
                 result.get("sanity_ok"),
                 json_mod.dumps(result.get("sanity_violations", [])),
                 result.get("json_path"),
                 result.get("csv_path"),
                 result.get("png_path"),
                 result.get("model_lat"),
                 result.get("model_lon"),
                 result.get("init_time_utc"),
                 result.get("mode_used"),
                 result.get("lon"), result.get("lat"),
                 job_id))
        elif status == "failed":
            cur.execute(
                "UPDATE forecast_runs SET status=%s, error_msg=%s WHERE id=%s",
                (status, error, job_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"DB update failed for job {job_id}: {e}")


def run_forecast_sync(
    job_id: str,
    location_name: str,
    lat: float,
    lon: float,
    days: int,
    mode: str,
    init_date: str = None,
) -> dict:
    """
    Run the full NeuralGCM forecast pipeline synchronously.
    Returns a serialisable result dict.
    """
    logger.info(
        f"Sync forecast started | job={job_id} | "
        f"loc={location_name} | mode={mode}")

    from neuralgcm_weather.pipeline.forecast import run_forecast_pipeline
    result = run_forecast_pipeline(
        location_name = location_name,
        lat           = lat,
        lon           = lon,
        forecast_days = days,
        init_date     = f"{init_date}T00:00" if init_date else None,
        mode          = mode,
        save          = True,
    )

    fp      = result["forecast_point"]
    elapsed = result["elapsed_seconds"]

    import numpy as np

    def _compass(deg):
        if deg is None:
            return None
        dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                "S","SSW","SW","WSW","W","WNW","NW","NNW"]
        return dirs[int(round(float(deg) + 11.25) // 22) % 16]

    def _stability(lr):
        if lr is None or np.isnan(lr):
            return None
        if lr > 9.8:
            return "UNSTABLE"
        if lr > 7.0:
            return "Conditionally unstable"
        return "Stable"

    daily = []
    for i, dt in enumerate(fp.dates):
        def _v(arr, i=i):
            if arr is None:
                return None
            v = arr[i]
            return None if np.isnan(v) else round(float(v), 4)

        wdir = _v(fp.wind_dir_850)
        lr   = _v(fp.lapse_rate)
        daily.append({
            "date": dt.strftime("%Y-%m-%d"),
            "temperature_c_850":    _v(fp.temperature_c_850),
            "temperature_c_500":    _v(fp.temperature_c_500),
            "rh_850":               _v(fp.rh_850),
            "rh_500":               _v(fp.rh_500),
            "specific_humidity_850": _v(fp.specific_humidity_850),
            "tpw_mm":               _v(fp.tpw_mm),
            "wind_speed_850":       _v(fp.wind_speed_850),
            "wind_speed_500":       _v(fp.wind_speed_500),
            "wind_speed_250":       _v(fp.wind_speed_250),
            "wind_dir_850":         wdir,
            "wind_dir_compass":     _compass(wdir),
            "u_850":                _v(fp.u_850),
            "v_850":                _v(fp.v_850),
            "z500_m":               _v(fp.z500_m),
            "mslp_hpa":             _v(fp.mslp_hpa),
            "lapse_rate":           lr,
            "stability":            _stability(lr),
            "clwc_gkg_850":         _v(fp.clwc_gkg_850),
            "ciwc_gkg_850":         _v(fp.ciwc_gkg_850),
            "vorticity_850":        _v(fp.vorticity_850),
        })

    output = {
        "job_id":           job_id,
        "status":           "complete",
        "location_name":    fp.location_name,
        "lat":              fp.lat,
        "lon":              fp.lon,
        "model_lat":        fp.model_lat,
        "model_lon":        fp.model_lon,
        "init_time_utc":    result["init_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode_used":        result["mode_used"],
        "forecast_days":    fp.days,
        "elapsed_seconds":  round(elapsed, 2),
        "is_cached":        False,
        "created_at":       datetime.utcnow().isoformat() + "Z",
        "daily":            daily,
        "sanity_ok":        len(result["violations"]) == 0,
        "sanity_violations": result["violations"],
        "json_path":        result["saved_files"].get("json"),
        "csv_path":         result["saved_files"].get("csv"),
        "png_path":         result["saved_files"].get("png"),
        "model_checkpoint": "v1/deterministic_2_8_deg.pkl",
        "paper_reference":  "Kochkov et al. 2024 (arXiv:2311.07222v3)",
    }

    logger.success(
        f"Sync forecast complete | job={job_id} | elapsed={elapsed:.1f}s")
    return output
