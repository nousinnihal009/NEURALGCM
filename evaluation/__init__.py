"""
End-to-end evaluation framework for meteorological forecast verification against ERA5.
"""
from .data_ingestion import load_era5_robustly, load_forecast
from .data_standardization import standardize_dataset
from .spatiotemporal_alignment import align_spatiotemporal
from .verification_metrics import calculate_metrics
from .diagnostics import compute_diagnostics
from .visualization import plot_spatial_errors, plot_timeseries
from .evaluation_pipeline import run_evaluation_pipeline

__all__ = [
    "load_era5_robustly",
    "load_forecast",
    "standardize_dataset",
    "align_spatiotemporal",
    "calculate_metrics",
    "compute_diagnostics",
    "plot_spatial_errors",
    "plot_timeseries",
    "run_evaluation_pipeline"
]
