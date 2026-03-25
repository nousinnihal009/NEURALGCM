"""
Forecast Plotter
================
Generates a multi-panel matplotlib dashboard from ForecastPoint data.
Dark theme matching the original forecast_anywhere.py aesthetics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
W      = "white"


def plot_forecast(fp, output_path: str, dpi: int = 150):
    """Generate a 6-panel forecast dashboard PNG."""
    N = fp.days + 1
    X = np.arange(N)
    date_labels = [dt.strftime("%a\n%d %b") for dt in fp.dates]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=DARK)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06,
                        right=0.96, hspace=0.35, wspace=0.25)

    # Title bar
    fig.text(0.5, 0.97, "NeuralGCM Weather Forecast",
             ha="center", va="top", color=W,
             fontsize=18, fontweight="bold")
    fig.text(0.5, 0.935,
             f"{fp.location_name}  |  "
             f"{fp.lat:.4f}°N, {fp.lon:.4f}°E  |  "
             f"{fp.dates[0].strftime('%d %b %Y')} → "
             f"{fp.dates[-1].strftime('%d %b %Y')}",
             ha="center", va="top", color="#8B949E", fontsize=10)

    def setup(ax, title, ylabel):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color="#58A6FF", fontsize=11,
                     fontweight="bold", pad=8, loc="left")
        ax.set_ylabel(ylabel, color=W, fontsize=9)
        ax.set_xticks(X)
        ax.set_xticklabels(date_labels, color=W, fontsize=8)
        ax.tick_params(axis="y", colors=W, labelsize=8)
        ax.grid(True, color=BORDER, ls="--", alpha=0.5, lw=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.set_xlim(-0.4, N - 0.6)

    # Panel 1: Temperature
    ax = axes[0, 0]
    if fp.temperature_c_850 is not None:
        ax.plot(X, fp.temperature_c_850, color="#F78166", lw=2.5,
                marker="o", ms=6, label="850 hPa")
        ax.fill_between(X, fp.temperature_c_850 - 2,
                        fp.temperature_c_850 + 2,
                        color="#F78166", alpha=0.12)
    if fp.temperature_c_500 is not None:
        ax.plot(X, fp.temperature_c_500, color="#EF9F27", lw=1.8,
                marker="s", ms=4, ls="--", label="500 hPa")
    setup(ax, "Temperature", "°C")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=W, edgecolor=BORDER)

    # Panel 2: Geopotential Height
    ax = axes[0, 1]
    if fp.z500_m is not None:
        ax.plot(X, fp.z500_m, color="#BD8EE6", lw=2.5,
                marker="o", ms=6, label="Z500")
    setup(ax, "Geopotential Height (500 hPa)", "m")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=W, edgecolor=BORDER)

    # Panel 3: Relative Humidity
    ax = axes[0, 2]
    if fp.rh_850 is not None:
        ax.bar(X - 0.15, fp.rh_850, 0.28, color="#58A6FF",
               alpha=0.8, label="850 hPa")
    if fp.rh_500 is not None:
        ax.bar(X + 0.15, fp.rh_500, 0.28, color="#1D9E75",
               alpha=0.75, label="500 hPa")
    ax.axhline(80, color="#F78166", lw=0.9, ls="--", alpha=0.6)
    ax.set_ylim(0, 115)
    setup(ax, "Relative Humidity", "%")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=W, edgecolor=BORDER)

    # Panel 4: Precipitable Water
    ax = axes[1, 0]
    if fp.tpw_mm is not None:
        ax.fill_between(X, 0, fp.tpw_mm, color="#A5D6FF", alpha=0.4)
        ax.plot(X, fp.tpw_mm, color="#A5D6FF", lw=2.5,
                marker="o", ms=6, label="TPW")
    ax.axhline(60, color="#F78166", lw=0.9, ls="--", alpha=0.6)
    ax.set_ylim(bottom=0)
    setup(ax, "Total Precipitable Water", "mm")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=W, edgecolor=BORDER)

    # Panel 5: Wind Speed
    ax = axes[1, 1]
    if fp.wind_speed_850 is not None:
        ax.plot(X, fp.wind_speed_850, color="#3FB950", lw=2.5,
                marker="o", ms=6, label="850 hPa")
    if fp.wind_speed_500 is not None:
        ax.plot(X, fp.wind_speed_500, color="#EF9F27", lw=2,
                marker="s", ms=4, ls="--", label="500 hPa")
    if fp.wind_speed_250 is not None:
        ax.plot(X, fp.wind_speed_250, color="#F78166", lw=1.5,
                marker="^", ms=4, ls=":", label="250 hPa (jet)")
    setup(ax, "Wind Speed", "m/s")
    ax.legend(fontsize=7, facecolor=PANEL, labelcolor=W, edgecolor=BORDER)

    # Panel 6: Surface Pressure
    ax = axes[1, 2]
    if fp.mslp_hpa is not None:
        ax.plot(X, fp.mslp_hpa, color="#58A6FF", lw=2.5,
                marker="o", ms=6, label="MSLP")
        ax.axhline(1013.25, color=W, lw=0.8, ls=":", alpha=0.4,
                   label="Std atm")
    elif fp.lapse_rate is not None:
        ax.plot(X, fp.lapse_rate, color="#BD8EE6", lw=2.5,
                marker="o", ms=6, label="Lapse Rate")
        ax.axhline(6.5, color=W, lw=0.8, ls=":", alpha=0.4,
                   label="Std atm 6.5°C/km")
        setup(ax, "Atmospheric Stability", "°C/km")
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=W,
                  edgecolor=BORDER)
    else:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                ha="center", va="center", color="#8B949E", fontsize=14)
    if fp.mslp_hpa is not None:
        setup(ax, "Surface Pressure", "hPa")
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=W,
                  edgecolor=BORDER)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=DARK)
    plt.close(fig)
    logger.success(f"Dashboard saved: {output_path}")
