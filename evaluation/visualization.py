"""
visualization.py
=================
Publication-quality scientific visualization for forecast verification.
Uses matplotlib + cartopy for geospatial accuracy.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Global Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

CMAP_ERROR = 'RdBu_r'
CMAP_ABS = 'YlOrRd'
CMAP_CORR = 'RdYlGn'


# ── Spatial Maps ─────────────────────────────────────────────────────────────

def plot_spatial_errors(
    metrics: Dict[str, Dict[str, xr.DataArray]],
    output_dir: str,
    variables: list[str] | None = None,
) -> list[str]:
    """
    Multi-panel spatial maps: Bias, RMSE, Correlation for each variable.
    Returns list of saved file paths.
    """
    saved = []
    if variables is None:
        variables = [v for v in metrics if not v.startswith('_')]

    for var in variables:
        m = metrics[var]
        fig, axes = plt.subplots(
            1, 3, figsize=(16, 4.5),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        panels = [
            ('bias', 'Mean Bias', CMAP_ERROR, True),
            ('rmse', 'RMSE', CMAP_ABS, False),
            ('pearson_r', 'Pearson r', CMAP_CORR, False),
        ]

        for ax, (key, title, cmap, diverging) in zip(axes, panels):
            data = m[key]
            ax.set_title(title, fontweight='bold')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')

            if diverging:
                vmax = float(np.abs(data).max())
                norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            else:
                norm = None

            im = ax.pcolormesh(
                data.longitude, data.latitude, data.values,
                transform=ccrs.PlateCarree(),
                cmap=cmap, norm=norm, shading='auto'
            )
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        fig.suptitle(f'Spatial Verification — {var}', fontsize=14, fontweight='bold', y=1.02)
        fpath = os.path.join(output_dir, f'spatial_{var}.png')
        fig.savefig(fpath)
        plt.close(fig)
        saved.append(fpath)
        logger.info(f"Saved: {fpath}")

    return saved


# ── Time Series ──────────────────────────────────────────────────────────────

def plot_timeseries(
    diagnostics: Dict[str, dict],
    output_dir: str,
    variables: list[str] | None = None,
) -> list[str]:
    """
    Multi-panel time series: spatial RMSE evolution, bias drift, and ACC.
    """
    saved = []
    if variables is None:
        variables = [v for v in diagnostics if not v.startswith('_')]

    for var in variables:
        d = diagnostics[var]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 1) Temporal RMSE
        ts_rmse = d['temporal_rmse']
        axes[0].plot(ts_rmse.time, ts_rmse.values, color='#d62728', linewidth=1.2)
        axes[0].fill_between(ts_rmse.time.values, 0, ts_rmse.values,
                             alpha=0.15, color='#d62728')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Spatial RMSE vs Time', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 2) Bias drift
        bd = d['bias_drift']
        axes[1].plot(bd.time, bd.values, color='#1f77b4', linewidth=1.2)
        axes[1].axhline(0, color='k', linewidth=0.5, linestyle='--')
        axes[1].set_ylabel('Bias (rolling mean)')
        axes[1].set_title('Bias Drift', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 3) Error decomposition pie-like bar (fraction systematic vs random)
        decomp = d['error_decomposition']
        sys_frac = float(decomp['systematic_fraction'].mean(skipna=True))
        rand_frac = 1.0 - sys_frac
        axes[2].barh(['Systematic (Bias²)', 'Random (Variance)'],
                     [sys_frac, rand_frac],
                     color=['#ff7f0e', '#2ca02c'], height=0.4)
        axes[2].set_xlim(0, 1)
        axes[2].set_xlabel('Fraction of Total MSE')
        axes[2].set_title('Error Decomposition', fontweight='bold')

        fig.suptitle(f'Temporal Diagnostics — {var}', fontsize=14,
                     fontweight='bold', y=1.01)
        fig.tight_layout()
        fpath = os.path.join(output_dir, f'timeseries_{var}.png')
        fig.savefig(fpath)
        plt.close(fig)
        saved.append(fpath)
        logger.info(f"Saved: {fpath}")

    return saved


# ── Distribution Comparison ──────────────────────────────────────────────────

def plot_distributions(
    fcst_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    output_dir: str,
    variables: list[str] | None = None,
) -> list[str]:
    """
    PDF / KDE comparison between forecast and observation distributions.
    """
    saved = []
    if variables is None:
        variables = list(set(fcst_ds.data_vars) & set(obs_ds.data_vars))

    for var in variables:
        fig, ax = plt.subplots(figsize=(8, 5))
        f_vals = fcst_ds[var].values.flatten()
        o_vals = obs_ds[var].values.flatten()
        f_vals = f_vals[~np.isnan(f_vals)]
        o_vals = o_vals[~np.isnan(o_vals)]

        bins = np.linspace(
            min(f_vals.min(), o_vals.min()),
            max(f_vals.max(), o_vals.max()),
            80
        )

        ax.hist(o_vals, bins=bins, density=True, alpha=0.45,
                label='ERA5', color='#1f77b4', edgecolor='none')
        ax.hist(f_vals, bins=bins, density=True, alpha=0.45,
                label='Forecast', color='#d62728', edgecolor='none')
        ax.set_xlabel(var)
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Distribution Comparison — {var}', fontweight='bold')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

        fpath = os.path.join(output_dir, f'dist_{var}.png')
        fig.savefig(fpath)
        plt.close(fig)
        saved.append(fpath)

    return saved


# ── Error Heatmap (Variable × Metric) ───────────────────────────────────────

def plot_metric_heatmap(
    metrics: Dict[str, Dict[str, xr.DataArray]],
    output_dir: str,
) -> str:
    """
    Summary heatmap: rows = variables, cols = scalar metrics.
    """
    variables = [v for v in metrics if not v.startswith('_')]
    scalar_keys = ['bias', 'mae', 'rmse']
    table = np.zeros((len(variables), len(scalar_keys)))

    for i, var in enumerate(variables):
        for j, key in enumerate(scalar_keys):
            table[i, j] = float(metrics[var][key].mean(skipna=True))

    fig, ax = plt.subplots(figsize=(7, max(3, len(variables) * 0.6)))
    im = ax.imshow(table, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(scalar_keys)))
    ax.set_xticklabels([k.upper() for k in scalar_keys])
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables)

    for i in range(len(variables)):
        for j in range(len(scalar_keys)):
            ax.text(j, i, f'{table[i, j]:.3f}', ha='center', va='center',
                    fontsize=9, color='white' if table[i, j] > table.max() * 0.6 else 'black')

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Verification Summary Heatmap', fontweight='bold')
    fpath = os.path.join(output_dir, 'metric_heatmap.png')
    fig.savefig(fpath)
    plt.close(fig)
    logger.info(f"Saved: {fpath}")
    return fpath


# ── Forecast vs ERA5 Side-by-Side Snapshot ───────────────────────────────────

def plot_snapshot(
    fcst_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    var: str,
    time_idx: int,
    output_dir: str,
) -> str:
    """
    Side-by-side map: Forecast | ERA5 | Difference at a single timestep.
    """
    f = fcst_ds[var].isel(time=time_idx)
    o = obs_ds[var].isel(time=time_idx)
    diff = f - o

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    for ax in axes:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')

    vmin = min(float(f.min()), float(o.min()))
    vmax = max(float(f.max()), float(o.max()))

    for ax, data, title in [
        (axes[0], f, 'Forecast'),
        (axes[1], o, 'ERA5'),
    ]:
        im = ax.pcolormesh(data.longitude, data.latitude, data.values,
                           transform=ccrs.PlateCarree(), cmap='viridis',
                           vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(title, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Difference panel with diverging colormap
    d_abs = float(np.abs(diff).max())
    im_d = axes[2].pcolormesh(diff.longitude, diff.latitude, diff.values,
                              transform=ccrs.PlateCarree(), cmap=CMAP_ERROR,
                              vmin=-d_abs, vmax=d_abs, shading='auto')
    axes[2].set_title('Difference (Fcst − ERA5)', fontweight='bold')
    fig.colorbar(im_d, ax=axes[2], shrink=0.8)

    time_str = str(fcst_ds.time.values[time_idx])[:19]
    fig.suptitle(f'{var} @ {time_str}', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fpath = os.path.join(output_dir, f'snapshot_{var}_t{time_idx}.png')
    fig.savefig(fpath)
    plt.close(fig)
    return fpath
