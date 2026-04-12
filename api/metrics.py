"""
NeuralGCM Prometheus Metrics
=============================
Custom metrics for the Grafana dashboard panels.
Imported by api/main.py and api/routers/forecast.py.
"""

from prometheus_client import Counter, Histogram, Gauge

# ── Forecast metrics ──────────────────────────────────────────
forecast_counter = Counter(
    "neuralgcm_forecasts_total",
    "Total number of NeuralGCM forecasts completed",
    ["mode", "status"],
)

inference_duration = Histogram(
    "neuralgcm_inference_duration_seconds",
    "NeuralGCM inference wall-clock time in seconds",
    buckets=[1, 2, 5, 10, 20, 30, 60, 90, 120, 180],
)

active_jobs = Gauge(
    "neuralgcm_active_jobs",
    "Currently running forecast jobs",
)

# ── Cache metrics ─────────────────────────────────────────────
cache_hits = Counter(
    "neuralgcm_cache_hits_total",
    "Forecast cache hits (50km proximity)",
)

cache_misses = Counter(
    "neuralgcm_cache_misses_total",
    "Forecast cache misses",
)

# ── Accuracy metrics (updated after verification) ─────────────
mae_t850 = Gauge(
    "neuralgcm_mae_temperature_850",
    "Mean absolute error T850 vs ERA5 (last 24h, Kelvin)",
)

mae_z500 = Gauge(
    "neuralgcm_mae_z500",
    "Mean absolute error Z500 vs ERA5 (last 24h, metres)",
)

# ── Location metrics ──────────────────────────────────────────
unique_locations = Gauge(
    "neuralgcm_unique_locations_24h",
    "Unique forecast locations in last 24 hours",
)
