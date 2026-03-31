"""
evaluation_pipeline.py
=======================
End-to-end orchestrator for the research-grade forecast evaluation framework.
Supports both programmatic and CLI / YAML-config invocation.
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .data_ingestion import load_era5_robustly, load_forecast
from .data_standardization import standardize_dataset
from .spatiotemporal_alignment import align_spatiotemporal
from .verification_metrics import calculate_metrics
from .diagnostics import compute_diagnostics
from .visualization import (
    plot_spatial_errors,
    plot_timeseries,
    plot_distributions,
    plot_metric_heatmap,
    plot_snapshot,
)

logger = logging.getLogger(__name__)


# ── Core Pipeline ────────────────────────────────────────────────────────────

def run_evaluation_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the full evaluation pipeline given a configuration dictionary.

    Parameters
    ----------
    config : dict
        Must contain at minimum:
        - era5_path: str
        - forecast_path: str
        - output_dir: str

        Optional:
        - forecast_format: str  (default 'netcdf')
        - variables: list[str]  (default: auto-detect shared vars)
        - bbox: dict with lat_min, lat_max, lon_min, lon_max
        - time_range: dict with start, end  (ISO strings)
        - snapshot_indices: list[int]

    Returns
    -------
    dict : pipeline results including paths to all saved outputs.
    """
    start = time.time()
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    _setup_logging(output_dir)

    logger.info("=" * 72)
    logger.info("  FORECAST EVALUATION PIPELINE — START")
    logger.info(f"  Timestamp : {datetime.utcnow().isoformat()}Z")
    logger.info("=" * 72)

    # ── Phase 1: Data Ingestion ──────────────────────────────────────────
    logger.info("▸ Phase 1: Data Ingestion")
    era5_ds = load_era5_robustly(config['era5_path'])
    fcst_ds = load_forecast(
        config['forecast_path'],
        fmt=config.get('forecast_format', 'netcdf'),
    )

    logger.info(f"  ERA5 shape : {dict(era5_ds.dims)}")
    logger.info(f"  Forecast shape: {dict(fcst_ds.dims)}")

    # ── Phase 2: Standardization ─────────────────────────────────────────
    logger.info("▸ Phase 2: Data Standardization")
    era5_ds = standardize_dataset(era5_ds)
    fcst_ds = standardize_dataset(fcst_ds)

    # ── Phase 3: Optional Clipping ───────────────────────────────────────
    bbox = config.get('bbox')
    if bbox:
        logger.info(f"▸ Clipping to bbox: {bbox}")
        era5_ds = _clip_bbox(era5_ds, bbox)
        fcst_ds = _clip_bbox(fcst_ds, bbox)

    time_range = config.get('time_range')
    if time_range:
        logger.info(f"▸ Slicing time: {time_range}")
        era5_ds = era5_ds.sel(time=slice(time_range['start'], time_range['end']))
        fcst_ds = fcst_ds.sel(time=slice(time_range['start'], time_range['end']))

    # ── Phase 4: Spatiotemporal Alignment ────────────────────────────────
    logger.info("▸ Phase 4: Spatiotemporal Alignment")
    fcst_ds, era5_ds = align_spatiotemporal(fcst_ds, era5_ds)
    logger.info(f"  Aligned dims: {dict(fcst_ds.dims)}")

    # ── Phase 5: Verification Metrics ────────────────────────────────────
    variables = config.get('variables')
    logger.info("▸ Phase 5: Verification Metrics")
    metrics = calculate_metrics(fcst_ds, era5_ds, variables=variables)

    # ── Phase 6: Diagnostics ─────────────────────────────────────────────
    logger.info("▸ Phase 6: Advanced Diagnostics")
    diagnostics = compute_diagnostics(fcst_ds, era5_ds, variables=variables)

    # ── Phase 7: Visualization ───────────────────────────────────────────
    logger.info("▸ Phase 7: Visualization")
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    saved_plots = []
    saved_plots += plot_spatial_errors(metrics, plots_dir, variables=variables)
    saved_plots += plot_timeseries(diagnostics, plots_dir, variables=variables)
    saved_plots += plot_distributions(fcst_ds, era5_ds, plots_dir, variables=variables)
    saved_plots.append(plot_metric_heatmap(metrics, plots_dir))

    # Optional snapshots
    snapshot_indices = config.get('snapshot_indices', [0])
    used_vars = variables or list(set(fcst_ds.data_vars) & set(era5_ds.data_vars))
    for var in used_vars[:3]:  # limit to first 3 vars for initial snapshot
        for idx in snapshot_indices:
            try:
                saved_plots.append(plot_snapshot(fcst_ds, era5_ds, var, idx, plots_dir))
            except Exception as e:
                logger.warning(f"Snapshot failed for {var}@t{idx}: {e}")

    # ── Phase 8: Save Structured Results ─────────────────────────────────
    logger.info("▸ Phase 8: Saving structured results")
    summary = _build_summary(metrics, diagnostics, used_vars)
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)

    csv_path = os.path.join(output_dir, 'metrics_table.csv')
    _save_csv(metrics, used_vars, csv_path)

    elapsed = time.time() - start
    logger.info(f"Pipeline completed in {elapsed:.1f}s")
    logger.info(f"Plots: {len(saved_plots)}  |  Output dir: {output_dir}")

    return {
        'metrics': metrics,
        'diagnostics': diagnostics,
        'saved_plots': saved_plots,
        'summary_path': summary_path,
        'csv_path': csv_path,
        'elapsed_seconds': elapsed,
    }


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Research-grade forecast verification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m evaluation.evaluation_pipeline \\
    --era5-path data/era5_june2020.grib \\
    --forecast-path data/forecast_june2020.nc \\
    --output-dir results/june2020 \\
    --variables t2m u10 v10

Or use a YAML config:
  python -m evaluation.evaluation_pipeline --config eval_config.yaml
"""
    )
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--era5-path', type=str)
    parser.add_argument('--forecast-path', type=str)
    parser.add_argument('--forecast-format', type=str, default='netcdf')
    parser.add_argument('--output-dir', type=str, default='./eval_output')
    parser.add_argument('--variables', nargs='+', default=None)
    parser.add_argument('--lat-min', type=float)
    parser.add_argument('--lat-max', type=float)
    parser.add_argument('--lon-min', type=float)
    parser.add_argument('--lon-max', type=float)
    parser.add_argument('--time-start', type=str)
    parser.add_argument('--time-end', type=str)

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        if not args.era5_path or not args.forecast_path:
            parser.error("--era5-path and --forecast-path are required when not using --config")
        config = {
            'era5_path': args.era5_path,
            'forecast_path': args.forecast_path,
            'forecast_format': args.forecast_format,
            'output_dir': args.output_dir,
            'variables': args.variables,
        }
        if all(v is not None for v in [args.lat_min, args.lat_max, args.lon_min, args.lon_max]):
            config['bbox'] = {
                'lat_min': args.lat_min,
                'lat_max': args.lat_max,
                'lon_min': args.lon_min,
                'lon_max': args.lon_max,
            }
        if args.time_start and args.time_end:
            config['time_range'] = {'start': args.time_start, 'end': args.time_end}

    run_evaluation_pipeline(config)


# ── Private Helpers ──────────────────────────────────────────────────────────

def _setup_logging(output_dir: str):
    log_path = os.path.join(output_dir, 'pipeline.log')
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode='w'),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=handlers,
        force=True,
    )


def _clip_bbox(ds: xr.Dataset, bbox: dict) -> xr.Dataset:
    return ds.sel(
        latitude=slice(bbox['lat_max'], bbox['lat_min']),
        longitude=slice(bbox['lon_min'], bbox['lon_max']),
    )


def _build_summary(
    metrics: Dict, diagnostics: Dict, variables: list[str]
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {'generated_at': datetime.utcnow().isoformat()}
    for var in variables:
        m = metrics.get(var, {})
        d = diagnostics.get(var, {})
        summary[var] = {
            'global_bias': float(m.get('bias', np.nan).mean(skipna=True)) if 'bias' in m else None,
            'global_rmse': float(m.get('rmse', np.nan).mean(skipna=True)) if 'rmse' in m else None,
            'global_mae': float(m.get('mae', np.nan).mean(skipna=True)) if 'mae' in m else None,
            'mean_acc': float(m.get('acc', np.nan).mean(skipna=True)) if 'acc' in m else None,
            'regional': d.get('regional', {}),
        }
    return summary


def _save_csv(metrics: Dict, variables: list[str], path: str):
    rows = []
    for var in variables:
        m = metrics.get(var, {})
        rows.append({
            'variable': var,
            'mean_bias': float(m.get('bias', np.nan).mean(skipna=True)) if 'bias' in m else np.nan,
            'mean_rmse': float(m.get('rmse', np.nan).mean(skipna=True)) if 'rmse' in m else np.nan,
            'mean_mae': float(m.get('mae', np.nan).mean(skipna=True)) if 'mae' in m else np.nan,
            'mean_pearson_r': float(m.get('pearson_r', np.nan).mean(skipna=True)) if 'pearson_r' in m else np.nan,
            'mean_acc': float(m.get('acc', np.nan).mean(skipna=True)) if 'acc' in m else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Saved metrics CSV: {path}")


if __name__ == '__main__':
    main()
