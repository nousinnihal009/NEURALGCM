"""
run_in_colab.py
================
Copy-paste-ready script for Google Colab.
Handles dependency installation, data paths, and runs the full pipeline.
"""
# ── Cell 1: Install Dependencies ─────────────────────────────────────────────
# !pip install -q xarray cfgrib eccodes matplotlib cartopy numpy pandas pyyaml

# ── Cell 2: Full Pipeline Runner ─────────────────────────────────────────────

import sys
import os

# Point Python to the evaluation package
# Adjust this path to where you uploaded / cloned the 'evaluation/' folder
# e.g. if you placed it at /content/evaluation/
EVAL_PACKAGE_DIR = '/content'
if EVAL_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, EVAL_PACKAGE_DIR)

from evaluation.evaluation_pipeline import run_evaluation_pipeline

# ── Configuration ────────────────────────────────────────────────────────────
config = {
    # Required: paths to your actual files on Colab
    'era5_path': '/content/drive/MyDrive/your_era5_file.grib',
    'forecast_path': '/content/drive/MyDrive/your_forecast.nc',
    'forecast_format': 'netcdf',  # Change to 'grib' if forecast is also GRIB

    # Output directory (created automatically)
    'output_dir': '/content/eval_output',

    # Variables to evaluate (None = auto-detect shared variables)
    # Set to specific list if you only want certain variables:
    # 'variables': ['temperature_2m', 'u_wind_10m', 'v_wind_10m'],
    'variables': None,

    # Optional: spatial bounding box
    # 'bbox': {
    #     'lat_min': 12.5,
    #     'lat_max': 13.5,
    #     'lon_min': 80.0,
    #     'lon_max': 80.5,
    # },

    # Optional: time range
    # 'time_range': {
    #     'start': '2020-06-25',
    #     'end': '2020-06-30',
    # },

    # Snapshot time indices for side-by-side plots
    'snapshot_indices': [0, 24, 72],
}

# ── Run ──────────────────────────────────────────────────────────────────────
results = run_evaluation_pipeline(config)

print(f"\n✅ Pipeline complete — {results['elapsed_seconds']:.1f}s")
print(f"   Summary JSON : {results['summary_path']}")
print(f"   CSV table    : {results['csv_path']}")
print(f"   Plots        : {len(results['saved_plots'])} files in {config['output_dir']}/plots/")

# ── Cell 3: Display Plots in Colab ───────────────────────────────────────────
from IPython.display import Image, display

for plot_path in results['saved_plots']:
    print(f"\n── {os.path.basename(plot_path)} ──")
    display(Image(filename=plot_path))
