"""
Evaluator -- Base vs Fine-Tuned on India 2024 Test Set
=======================================================
Evaluates both checkpoints on Dec 2024 test data.
Metrics: RMSE, MAE, % improvement for T850,T500,Z500,Q850,U850,V850
at lead times 6h, 24h, 48h, 72h, 96h, 120h.
"""

import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
_ = jax.devices()

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger
from typing import List, Optional

import neuralgcm
from neuralgcm_weather.model.runner import (
    build_regridder, regrid_init_state, run_forecast)
from dinosaur import xarray_utils

# Dark theme colours
DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
W      = "white"

EVAL_VARS = [
    {"var": "temperature",         "level": 850,
     "name": "T850", "unit": "K",    "scale": 1.0},
    {"var": "temperature",         "level": 500,
     "name": "T500", "unit": "K",    "scale": 1.0},
    {"var": "geopotential",        "level": 500,
     "name": "Z500", "unit": "m",    "scale": 1 / 9.80665},
    {"var": "specific_humidity",   "level": 850,
     "name": "Q850", "unit": "g/kg", "scale": 1000.0},
    {"var": "u_component_of_wind", "level": 850,
     "name": "U850", "unit": "m/s",  "scale": 1.0},
    {"var": "v_component_of_wind", "level": 850,
     "name": "V850", "unit": "m/s",  "scale": 1.0},
]
LEAD_HOURS = [6, 24, 48, 72, 96, 120]


class IndiaEvaluator:
    """
    Evaluates base and fine-tuned NeuralGCM models on the held-out
    December 2024 test set over the Indian subcontinent.
    """

    def __init__(self, config, loader, test_times):
        self.config     = config
        self.loader     = loader
        self.test_times = test_times[:40]  # cap for speed

    def run(self) -> pd.DataFrame:
        """Run full evaluation, return results DataFrame."""
        base_ckpt_path = self.config["model"].get(
            "existing_checkpoint_cache",
            "cache/v1_deterministic_2_8_deg.pkl")
        ft_path = (
            "neuralgcm_finetune/checkpoints/best/"
            "finetuned_india_2024.pkl")

        # Load base model
        logger.info("Loading base model...")
        base_model = self._load_model(base_ckpt_path)

        # Load fine-tuned model
        ft_model = None
        if Path(ft_path).exists():
            logger.info("Loading fine-tuned model...")
            try:
                ft_model = self._load_model(ft_path)
            except Exception as e:
                logger.warning(
                    f"Could not load fine-tuned model: {e}")
                logger.info("Evaluating base model only")

        if base_model is None:
            logger.error("Could not load base model")
            return pd.DataFrame()

        press_v = self.config["model"]["input_vars"]
        surf_v  = self.config["model"]["surface_vars"]
        ts_h    = self.config["model"]["timestep_hours"]
        rows    = []

        for init_t in self.test_times:
            logger.info(
                f"  Evaluating "
                f"{init_t.strftime('%Y-%m-%d %H:%M')}")
            try:
                init_ds = self.loader.get_init_state(
                    init_t, press_v, surf_v)

                # Run base model forecast
                base_ds = self._run_model_forecast(
                    base_model, init_ds, ts_h)

                # Run fine-tuned model forecast
                ft_ds = None
                if ft_model is not None:
                    ft_ds = self._run_model_forecast(
                        ft_model, init_ds, ts_h)

                for lead_h in LEAD_HOURS:
                    tgt_t = init_t + pd.Timedelta(hours=lead_h)
                    nearest = min(
                        self.test_times,
                        key=lambda t: abs(
                            (t - tgt_t).total_seconds()))
                    if abs((nearest - tgt_t).total_seconds()) > \
                            3600 * ts_h:
                        continue

                    truth_ds = self.loader.get_init_state(
                        nearest, press_v, surf_v)
                    step_i = lead_h // ts_h

                    for vc in EVAL_VARS:
                        b_arr = self._extract_field(
                            base_ds, vc["var"],
                            vc["level"], step_i)
                        t_arr = self._extract_field(
                            truth_ds, vc["var"],
                            vc["level"], 0)

                        if b_arr is None or t_arr is None:
                            continue

                        sc = vc["scale"]
                        row = {
                            "init_time":  str(init_t),
                            "lead_hours": lead_h,
                            "variable":   vc["name"],
                            "unit":       vc["unit"],
                            "base_rmse":  self._rmse(
                                b_arr * sc, t_arr * sc),
                            "base_mae":   self._mae(
                                b_arr * sc, t_arr * sc),
                        }

                        if ft_ds is not None:
                            f_arr = self._extract_field(
                                ft_ds, vc["var"],
                                vc["level"], step_i)
                            if f_arr is not None:
                                row["ft_rmse"] = self._rmse(
                                    f_arr * sc, t_arr * sc)
                                row["ft_mae"] = self._mae(
                                    f_arr * sc, t_arr * sc)
                                row["pct_improve"] = (
                                    100 * (row["base_rmse"] -
                                           row["ft_rmse"]) /
                                    max(row["base_rmse"], 1e-10))
                            else:
                                row["ft_rmse"] = None
                                row["ft_mae"] = None
                                row["pct_improve"] = None
                        else:
                            row["ft_rmse"] = None
                            row["ft_mae"] = None
                            row["pct_improve"] = None

                        rows.append(row)

            except Exception as e:
                logger.warning(
                    f"  Failed for {init_t}: {e}")

        df = pd.DataFrame(rows)
        logger.info(f"Evaluation complete: {len(df)} rows")
        return df

    def _load_model(self, path: str):
        """Load a NeuralGCM model from checkpoint file."""
        p = Path(path)
        if not p.exists():
            logger.error(f"Checkpoint not found: {p}")
            return None

        try:
            with open(p, "rb") as f:
                ckpt = pickle.load(f)

            # Handle fine-tuned checkpoint format
            if isinstance(ckpt, dict) and "base_checkpoint" in ckpt:
                ckpt = ckpt["base_checkpoint"]

            return neuralgcm.PressureLevelModel.from_checkpoint(
                ckpt)
        except Exception as e:
            logger.error(f"Failed to load model from {p}: {e}")
            return None

    def _run_model_forecast(
        self, model, init_ds, ts_h,
        max_lead_h: int = 120,
    ) -> Optional[object]:
        """Run a forecast with the given model."""
        try:
            regridder = build_regridder(init_ds, model)
            ev = regrid_init_state(init_ds, regridder)
            forecast_days = max_lead_h // 24
            ds, _ = run_forecast(
                model, ev,
                forecast_days=forecast_days,
                timestep_hours=ts_h)
            return ds
        except Exception as e:
            logger.debug(f"Forecast failed: {e}")
            return None

    def _extract_field(
        self, ds, var: str, level: int,
        step_idx: int,
    ) -> Optional[np.ndarray]:
        """Extract a 2D field from forecast dataset."""
        if ds is None or var not in ds.data_vars:
            return None
        try:
            da = ds[var]
            time_dim = next(
                (d for d in da.dims
                 if "time" in d.lower() or
                 "delta" in d.lower()),
                None)
            if time_dim:
                idx = min(step_idx,
                          da.sizes[time_dim] - 1)
                da = da.isel({time_dim: idx})
            if level and "level" in da.dims:
                da = da.sel(level=level, method="nearest")
            return np.array(da).squeeze()
        except Exception:
            return None

    def _rmse(self, p: np.ndarray, t: np.ndarray) -> float:
        """Area-weighted RMSE."""
        nlat = p.shape[-2] if p.ndim >= 2 else 1
        lats = np.deg2rad(np.linspace(90, -90, nlat))
        w = np.cos(lats) / np.cos(lats).mean()
        diff = (p - t) ** 2
        if p.ndim >= 2:
            weighted = diff * w[:, np.newaxis]
        else:
            weighted = diff
        return float(np.sqrt(np.mean(weighted)))

    def _mae(self, p: np.ndarray, t: np.ndarray) -> float:
        """Mean absolute error."""
        return float(np.mean(np.abs(p - t)))

    def plot(self, df: pd.DataFrame):
        """Generate RMSE comparison plots."""
        if df.empty:
            logger.warning("No results to plot")
            return

        out = Path("neuralgcm_finetune/results")
        out.mkdir(parents=True, exist_ok=True)

        vars_ = df["variable"].unique().tolist()
        has_ft = df["ft_rmse"].notna().any()

        nrows = len(vars_)
        ncols = 2 if has_ft else 1
        fig = plt.figure(
            figsize=(7 * ncols, 4 * nrows),
            facecolor=DARK)
        fig.suptitle(
            "NeuralGCM Fine-Tuned on India ERA5 2024\n"
            "Base vs Fine-Tuned RMSE - Indian Subcontinent",
            color=W, fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(
            nrows, ncols, hspace=0.5, wspace=0.3,
            left=0.09, right=0.96, top=0.91, bottom=0.05)

        for ri, var in enumerate(vars_):
            sub = df[df["variable"] == var]
            if sub.empty:
                continue

            g = sub.groupby("lead_hours").agg(
                base_rmse=("base_rmse", "mean"),
            ).reset_index()

            if has_ft:
                ft_agg = sub.dropna(subset=["ft_rmse"]).groupby(
                    "lead_hours").agg(
                    ft_rmse=("ft_rmse", "mean"),
                    pct=("pct_improve", "mean"),
                ).reset_index()
                g = g.merge(ft_agg, on="lead_hours", how="left")

            unit = sub["unit"].iloc[0]

            # RMSE plot
            ax = fig.add_subplot(gs[ri, 0])
            ax.set_facecolor(PANEL)
            ax.plot(
                g["lead_hours"], g["base_rmse"],
                color="#8B949E", lw=2.5, marker="o",
                ms=5, label="Base")
            if has_ft and "ft_rmse" in g.columns:
                ax.plot(
                    g["lead_hours"],
                    g["ft_rmse"],
                    color="#58A6FF", lw=2.5, marker="s",
                    ms=5, label="Fine-tuned (India 2024)")
            ax.set_title(
                f"{var} RMSE", color="#58A6FF",
                fontsize=10, fontweight="bold")
            ax.set_xlabel(
                "Lead time (h)", color=W, fontsize=8)
            ax.set_ylabel(
                f"RMSE ({unit})", color=W, fontsize=8)
            ax.tick_params(colors=W, labelsize=8)
            ax.grid(True, color=BORDER, ls="--", alpha=0.5)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.legend(
                fontsize=7, facecolor=PANEL,
                labelcolor=W, edgecolor=BORDER)

            # Improvement bar chart
            if has_ft and ncols > 1 and "pct" in g.columns:
                ax2 = fig.add_subplot(gs[ri, 1])
                ax2.set_facecolor(PANEL)
                pct_vals = g["pct"].fillna(0)
                cols = [
                    "#3FB950" if v > 0 else "#F78166"
                    for v in pct_vals]
                ax2.bar(
                    g["lead_hours"], pct_vals,
                    color=cols, alpha=0.8, width=6)
                ax2.axhline(0, color=W, lw=0.8, alpha=0.4)
                ax2.set_title(
                    f"{var} Improvement (%)",
                    color="#58A6FF", fontsize=10,
                    fontweight="bold")
                ax2.set_xlabel(
                    "Lead time (h)", color=W, fontsize=8)
                ax2.set_ylabel(
                    "% RMSE change (+green=better)",
                    color=W, fontsize=8)
                ax2.tick_params(colors=W, labelsize=8)
                ax2.grid(
                    True, color=BORDER, ls="--", alpha=0.4)
                for sp in ax2.spines.values():
                    sp.set_edgecolor(BORDER)

        plt.savefig(
            out / "rmse_comparison.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close()
        logger.success(
            "Plot saved: "
            "neuralgcm_finetune/results/rmse_comparison.png")

    def print_summary(self, df: pd.DataFrame):
        """Print evaluation summary table."""
        if df.empty:
            logger.warning("No results to summarise")
            return

        has_ft = df["ft_rmse"].notna().any()

        print("\n" + "=" * 68)
        print("  INDIA 2024 EVALUATION RESULTS")
        print("  Region: 6N-37N, 68E-97E")
        print("  Test:   December 2024 (held out)")
        print("=" * 68)

        if has_ft:
            print(f"  {'Var':<6} {'Lead':>5}h "
                  f"{'Base':>10} {'FT':>10} {'Pct':>7}")
            print("  " + "-" * 46)
        else:
            print(f"  {'Var':<6} {'Lead':>5}h "
                  f"{'Base RMSE':>12}")
            print("  " + "-" * 30)

        for lead in sorted(df["lead_hours"].unique()):
            for var in sorted(df["variable"].unique()):
                sub = df[(df["lead_hours"] == lead) &
                         (df["variable"] == var)]
                if sub.empty:
                    continue
                b = sub["base_rmse"].mean()
                u = sub["unit"].iloc[0]
                if has_ft and sub["ft_rmse"].notna().any():
                    f = sub["ft_rmse"].mean()
                    d = sub["pct_improve"].mean()
                    s = "up" if d > 0 else "dn"
                    print(
                        f"  {var:<6} {lead:>5}h  "
                        f"{b:>7.4f} {u}  "
                        f"{f:>7.4f} {u}  "
                        f"{s}{abs(d):>5.1f}%")
                else:
                    print(
                        f"  {var:<6} {lead:>5}h  "
                        f"{b:>7.4f} {u}")

        if has_ft and df["pct_improve"].notna().any():
            ov = df["pct_improve"].dropna().mean()
            print("=" * 68)
            print(f"  Overall improvement: {ov:+.2f}%")

        print("=" * 68)
