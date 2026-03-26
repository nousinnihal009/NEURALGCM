"""
10-Panel NeuralGCM Weather Dashboard
=====================================
Dark-themed matplotlib dashboard matching original forecast_anywhere.py
panel count. All panels labelled with units, thresholds, and location.
ERA5 ground-truth overlay is drawn when era5_truth dict is provided.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings("ignore")

BG     = "#0d1117"
FG     = "#e6edf3"
ACCENT = "#58a6ff"
WARN   = "#f85149"
OK     = "#3fb950"


def plot_forecast(fp, output_path: str, era5_truth: dict = None,
                  dpi: int = 150) -> None:
    """
    Generate a 10-panel weather dashboard PNG.

    Args:
        fp:           ForecastPoint object (NeuralGCM output)
        output_path:  path to save PNG
        era5_truth:   optional dict mapping variable name → np.ndarray
                      of ERA5 ground-truth values for overlay
        dpi:          output resolution (default 150)
    """
    days = np.arange(fp.days + 1)
    dates_short = [d.strftime("%a\n%d %b") for d in fp.dates]

    fig = plt.figure(figsize=(22, 18), facecolor=BG)
    gs  = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        left=0.06, right=0.97,
        top=0.92,  bottom=0.06,
    )

    axes = [fig.add_subplot(gs[r, c])
            for r in range(3) for c in range(3)]
    # Panel 10 (world map) spans bottom row
    ax_map = fig.add_subplot(gs[3, :])

    def style(ax, title, ylabel, unit=""):
        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=9, pad=4,
                     fontweight="bold")
        ax.set_ylabel(f"{ylabel} ({unit})" if unit else ylabel,
                      color=FG, fontsize=7)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.set_xticks(days)
        ax.set_xticklabels(dates_short, fontsize=6.5)
        ax.grid(axis="y", color="#21262d", linewidth=0.5)

    def plot_var(ax, arr, color, label="", truth=None,
                 hline=None, hline_label=""):
        if arr is None:
            ax.text(0.5, 0.5, "N/A", color="#888", ha="center",
                    va="center", transform=ax.transAxes)
            return
        ax.plot(days, arr, color=color, lw=2, marker="o",
                ms=4, label=label)
        if truth is not None:
            ax.plot(days, truth, color="white", lw=1.5,
                    linestyle="--", alpha=0.7, label="ERA5 truth")
            ax.legend(fontsize=6, facecolor="#161b22",
                      labelcolor=FG, framealpha=0.8)
        if hline is not None:
            ax.axhline(hline, color=WARN, lw=0.8,
                       linestyle=":", alpha=0.7)
            ax.text(days[-1], hline, f" {hline_label}",
                    color=WARN, fontsize=6, va="bottom")

    truth = era5_truth or {}

    # ── Panel 1: Temperature 850 hPa ─────────────────────────
    ax = axes[0]
    plot_var(ax, fp.temperature_c_850, ACCENT, "T@850hPa",
             truth=truth.get("temperature_c_850"))
    style(ax, "Temperature @ 850 hPa", "°C")

    # ── Panel 2: Relative Humidity 850 hPa ───────────────────
    ax = axes[1]
    plot_var(ax, fp.rh_850, "#79c0ff", "RH@850hPa",
             truth=truth.get("rh_850"), hline=70, hline_label="70%")
    style(ax, "Relative Humidity @ 850 hPa", "%")
    ax.set_ylim(0, 105)

    # ── Panel 3: Total Precipitable Water ────────────────────
    ax = axes[2]
    plot_var(ax, fp.tpw_mm, "#56d364", "TPW",
             truth=truth.get("tpw_mm"), hline=60, hline_label="60mm")
    style(ax, "Total Precipitable Water", "mm")

    # ── Panel 4: Wind Speed 850 / 500 / 250 hPa ──────────────
    ax = axes[3]
    for arr, col, lbl in [
        (fp.wind_speed_850, ACCENT,   "850 hPa"),
        (fp.wind_speed_500, "#d2a8ff","500 hPa"),
        (fp.wind_speed_250, "#ff7b72","250 hPa"),
    ]:
        if arr is not None:
            ax.plot(days, arr, color=col, lw=1.8,
                    marker="o", ms=3, label=lbl)
    ax.legend(fontsize=6, facecolor="#161b22",
              labelcolor=FG, framealpha=0.8)
    style(ax, "Wind Speed (3 levels)", "m/s")

    # ── Panel 5: Geopotential Height Z500 ────────────────────
    ax = axes[4]
    plot_var(ax, fp.z500_m, "#ffa657", "Z500",
             truth=truth.get("z500_m"))
    style(ax, "Geopotential Height Z500", "m")

    # ── Panel 6: Surface Pressure ─────────────────────────────
    ax = axes[5]
    plot_var(ax, fp.mslp_hpa, "#ff7b72", "SP",
             truth=truth.get("mslp_hpa"))
    style(ax, "Surface Pressure", "hPa")

    # ── Panel 7: Cloud Water Content (850 hPa) ───────────────
    ax = axes[6]
    if fp.clwc_gkg_850 is not None:
        ax.bar(days, fp.clwc_gkg_850, color=ACCENT,
               alpha=0.7, label="Liquid")
    if fp.ciwc_gkg_850 is not None:
        ax.bar(days, fp.ciwc_gkg_850, color="#d2a8ff",
               alpha=0.7, label="Ice", bottom=fp.clwc_gkg_850
               if fp.clwc_gkg_850 is not None else None)
    ax.legend(fontsize=6, facecolor="#161b22",
              labelcolor=FG, framealpha=0.8)
    style(ax, "Cloud Water @ 850 hPa (Liquid + Ice)", "g/kg")

    # ── Panel 8: Lapse Rate 850→500 hPa ──────────────────────
    ax = axes[7]
    plot_var(ax, fp.lapse_rate, "#ffa657", "Lapse rate",
             hline=9.8, hline_label="DALR")
    if fp.lapse_rate is not None:
        ax.axhline(6.5, color=OK, lw=0.8, linestyle=":",
                   alpha=0.7)
        ax.text(days[-1], 6.5, " MALR", color=OK,
                fontsize=6, va="bottom")
    style(ax, "Lapse Rate 850→500 hPa", "°C/km")

    # ── Panel 9: Vorticity 850 hPa ───────────────────────────
    ax = axes[8]
    if fp.vorticity_850 is not None:
        colors = [WARN if v > 0 else ACCENT
                  for v in fp.vorticity_850]
        ax.bar(days, fp.vorticity_850, color=colors, alpha=0.8)
        ax.axhline(0, color=FG, lw=0.6, linestyle="-")
    else:
        ax.text(0.5, 0.5, "N/A", color="#888", ha="center",
                va="center", transform=ax.transAxes)
    style(ax, "Relative Vorticity @ 850 hPa", "1/s × 10⁻⁵")

    # ── Panel 10: World Map with Location Marker ─────────────
    ax_map.set_facecolor("#0d1f2d")
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-90, 90)
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        # If cartopy available, use it
        ax_map.remove()
        ax_map = fig.add_subplot(gs[3, :],
                                 projection=ccrs.PlateCarree())
        ax_map.set_facecolor("#0d1f2d")
        ax_map.add_feature(cfeature.LAND,   facecolor="#161b22",
                           edgecolor="#30363d", linewidth=0.4)
        ax_map.add_feature(cfeature.OCEAN,  facecolor="#0d1f2d")
        ax_map.add_feature(cfeature.BORDERS,edgecolor="#21262d",
                           linewidth=0.3)
        ax_map.set_global()
    except ImportError:
        # Fallback: plain axes with coast approximation
        ax_map.set_xlabel("Longitude", color=FG, fontsize=8)
        ax_map.set_ylabel("Latitude",  color=FG, fontsize=8)
        ax_map.tick_params(colors=FG, labelsize=7)
        for sp in ax_map.spines.values():
            sp.set_color("#30363d")
        ax_map.grid(color="#21262d", linewidth=0.4)

    # Plot location marker
    ax_map.plot(fp.lon, fp.lat, "o", color=WARN,
                ms=8, zorder=5, transform=getattr(
                    ax_map, "transData", None) or ax_map.transData)
    for r, alpha in [(5, 0.15), (10, 0.08), (20, 0.04)]:
        circle = Circle((fp.lon, fp.lat), r,
                         color=WARN, fill=True,
                         alpha=alpha, zorder=4,
                         transform=getattr(
                             ax_map, "transData", None)
                         or ax_map.transData)
        ax_map.add_patch(circle)
    ax_map.annotate(
        f" {fp.location_name}\n ({fp.lat:.2f}°N, {fp.lon:.2f}°E)",
        xy=(fp.lon, fp.lat),
        color=FG, fontsize=8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#161b22", edgecolor=ACCENT,
                  alpha=0.9),
    )
    ax_map.set_title("Forecast Location", color=FG,
                     fontsize=9, fontweight="bold")

    # ── Figure title ──────────────────────────────────────────
    init_str = fp.dates[0].strftime("%d %B %Y %H:%M UTC")
    fig.suptitle(
        f"NeuralGCM  ·  {fp.location_name}  ·  "
        f"Init: {init_str}  ·  "
        f"{fp.days}-Day Forecast",
        color=FG, fontsize=13, fontweight="bold", y=0.975,
    )

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    from loguru import logger
    logger.success(f"10-panel dashboard saved: {output_path}")
