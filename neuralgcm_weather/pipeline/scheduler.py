"""
Automated Forecast Scheduler
==============================
Runs NeuralGCM every 6 hours for all configured locations.
Uses APScheduler. Saves all results to disk automatically.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger
import pandas as pd


def run_all_locations():
    """Run forecasts for all locations in config.yaml."""
    from neuralgcm_weather.config import CFG_RAW
    from neuralgcm_weather.pipeline.forecast import run_forecast_pipeline

    locations = CFG_RAW["scheduler"]["locations"]
    logger.info(
        f"Scheduled run starting | "
        f"{len(locations)} locations | "
        f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    results = []
    for loc in locations:
        try:
            result = run_forecast_pipeline(
                location_name = loc["name"],
                lat           = loc["lat"],
                lon           = loc["lon"],
                mode          = "realtime",
            )
            results.append({
                "location": loc["name"],
                "status":   "ok",
                "elapsed":  result["elapsed_seconds"],
                "files":    result["saved_files"],
            })
        except Exception as e:
            logger.error(
                f"Failed for {loc['name']}: {e}")
            results.append({
                "location": loc["name"],
                "status": "error",
                "error":  str(e),
            })

    ok    = sum(1 for r in results if r["status"] == "ok")
    total = len(results)
    logger.success(
        f"Scheduled run complete | {ok}/{total} locations succeeded")
    return results


def start_scheduler(blocking: bool = True):
    """
    Start the APScheduler that runs forecasts every 6 hours.
    Set blocking=False for use inside other applications.
    """
    from neuralgcm_weather.config import CFG_RAW
    interval = CFG_RAW["scheduler"].get("interval_hours", 6)

    SchedulerClass = (BlockingScheduler if blocking
                      else BackgroundScheduler)
    scheduler = SchedulerClass()

    scheduler.add_job(
        run_all_locations,
        trigger  = "interval",
        hours    = interval,
        id       = "neuralgcm_forecast",
        name     = f"NeuralGCM forecast every {interval}h",
        replace_existing = True,
        misfire_grace_time = 3600,
    )

    logger.info(
        f"Scheduler configured | "
        f"interval={interval}h | "
        f"locations={len(CFG_RAW['scheduler']['locations'])}")

    # Run once immediately on start
    logger.info("Running initial forecast now...")
    run_all_locations()

    logger.info(
        f"Starting scheduler (next run in {interval}h)...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()
