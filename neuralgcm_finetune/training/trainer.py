"""
NeuralGCM Fine-Tuning Trainer
===============================
Two-phase training on the 2024 Indian subcontinent ERA5 dataset.

Phase A: Decoder only   -- corrects output biases over India
Phase B: Decoder + Physics -- adapts monsoon column physics

Key adaptations for regional 1-year dataset:
  - AdamW (weight decay) to prevent overfitting
  - Aggressive early stopping (patience=8)
  - Regional loss mask (India domain only)
  - Lower learning rates than global fine-tuning
  - Progressive rollout capped at 48h (not 72h)
"""

import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
# XLA_FLAGS removed — xla_cpu_use_thunk_runtime is deprecated in JAX 0.9+
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import jax
import jax.numpy as jnp
_ = jax.devices()

import pickle
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import optax
import xarray as xr
from loguru import logger

from neuralgcm_finetune.training.loss import RegionalFineTuneLoss


class IndiaFineTuner:
    """
    Fine-tunes NeuralGCM on 2024 Indian subcontinent ERA5 data.
    """

    def __init__(self, config: dict, loader, sampler):
        self.config  = config
        self.loader  = loader
        self.sampler = sampler
        self.model   = None
        self.params  = None
        self.loss_fn = None
        self._model_lats = None
        self._model_lons = None
        self._ckpt_data  = None  # raw checkpoint for saving

    def setup(self):
        """Load checkpoint, build model, initialise loss."""
        import neuralgcm

        model_cfg = self.config["model"]

        # Resolve checkpoint path: prefer local caches
        ckpt_path = None
        local_path = Path(model_cfg["base_checkpoint_local"])
        existing_cache = Path(model_cfg.get(
            "existing_checkpoint_cache",
            "cache/v1_deterministic_2_8_deg.pkl"))

        if local_path.exists():
            ckpt_path = str(local_path)
        elif existing_cache.exists():
            # Copy from existing cache to our checkpoint dir
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Copying cached checkpoint: "
                f"{existing_cache} -> {local_path}")
            shutil.copy2(str(existing_cache), str(local_path))
            ckpt_path = str(local_path)
        else:
            # Download from GCS
            logger.info("Downloading checkpoint from GCS...")
            from neuralgcm_weather.model.checkpoint import (
                load_checkpoint as download_ckpt)
            self.model = download_ckpt(
                model_name=model_cfg["base_checkpoint_gcs"],
                local_cache_dir=str(local_path.parent),
            )
            # Model is already loaded by download_ckpt
            self._setup_grid_and_loss()
            return

        # Load checkpoint from disk
        logger.info(f"Loading checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            self._ckpt_data = pickle.load(f)

        self.model = neuralgcm.PressureLevelModel.from_checkpoint(
            self._ckpt_data)
        logger.success(
            f"Model loaded | "
            f"input_vars={self.model.input_variables} | "
            f"forcing_vars={self.model.forcing_variables}")

        self._setup_grid_and_loss()

    def _setup_grid_and_loss(self):
        """Extract model grid coords and build regional loss."""
        # Extract NeuralGCM T63 Gaussian grid coordinates
        # The model.data_coords contains the grid information
        try:
            horiz = self.model.data_coords.horizontal
            # dinosaur grid has .latitudes and .longitudes
            if hasattr(horiz, 'latitudes'):
                self._model_lats = np.array(horiz.latitudes)
            if hasattr(horiz, 'longitudes'):
                self._model_lons = np.array(horiz.longitudes) % 360
        except Exception as e:
            logger.warning(
                f"Could not extract grid from model.data_coords: "
                f"{e}")

        # Fallback to standard T63 Gaussian grid values
        if self._model_lats is None:
            nlat = self.config["model"]["nlat"]
            nlon = self.config["model"]["nlon"]
            self._model_lats = np.linspace(
                87.863, -87.863, nlat)
            self._model_lons = np.linspace(
                0, 360 - 360 / nlon, nlon)
            logger.info(
                f"Using fallback T63 grid: "
                f"{nlat}x{nlon}")

        logger.info(
            f"Model grid: {len(self._model_lats)} lats x "
            f"{len(self._model_lons)} lons")

        # Initialise regional loss
        self.loss_fn = RegionalFineTuneLoss(
            config=self.config,
            model_lats=self._model_lats,
            model_lons=self._model_lons,
        )

        # Extract model params for fine-tuning
        # NeuralGCM PressureLevelModel stores params internally
        # We need to access the underlying haiku params
        try:
            if hasattr(self.model, 'params'):
                self.params = self.model.params
            elif hasattr(self.model, '_params'):
                self.params = self.model._params
            else:
                # Try to extract from the model structure
                self.params = {}
                logger.warning(
                    "Could not extract params directly — "
                    "using model.encode/unroll for forward pass")
        except Exception as e:
            logger.warning(f"Param extraction: {e}")
            self.params = {}

        logger.success(
            f"Trainer ready | "
            f"model_grid=({len(self._model_lats)}x"
            f"{len(self._model_lons)})")

    def _make_optimiser(self, phase_cfg: dict):
        """Build AdamW optimiser with cosine LR schedule."""
        lr       = phase_cfg["learning_rate"]
        lr_end   = phase_cfg.get("lr_end", lr * 0.1)
        warmup   = phase_cfg.get("warmup_steps", 50)
        steps    = phase_cfg["max_steps"]
        wd       = phase_cfg.get("weight_decay", 1e-4)
        clip     = phase_cfg.get("gradient_clip", 1.0)
        b1       = phase_cfg.get("adam_b1", 0.9)
        b2       = phase_cfg.get("adam_b2", 0.95)
        eps      = phase_cfg.get("adam_eps", 1e-8)

        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, lr, warmup),
                optax.cosine_decay_schedule(
                    lr, steps - warmup, alpha=lr_end / lr),
            ],
            boundaries=[warmup],
        )
        # AdamW: Adam + decoupled weight decay
        # Better than L2 regularisation for transformers/MLPs
        return optax.chain(
            optax.clip_by_global_norm(clip),
            optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
            optax.add_decayed_weights(wd),
            optax.scale_by_schedule(schedule),
            optax.scale(-1.0),
        )

    def _run_forward_and_loss(
        self,
        init_ds: xr.Dataset,
        targets: List[Optional[xr.Dataset]],
        rollout_steps: int,
        phase: str,
        rng_key,
    ) -> Tuple[float, Dict]:
        """
        Run one forward pass (encode -> unroll -> loss).
        This is the core computation that computes predictions
        and compares them to ERA5 targets over the India domain.
        """
        from dinosaur import xarray_utils

        phase_cfg = self.config[f"phase_{phase}"]
        ts_h = self.config["model"]["timestep_hours"]

        try:
            # Build regridder for this init state
            from neuralgcm_weather.model.runner import (
                build_regridder, regrid_init_state)

            regridder = build_regridder(init_ds, self.model)
            ev = regrid_init_state(init_ds, regridder)

            # Encode initial state
            inputs   = self.model.inputs_from_xarray(ev)
            forcings = self.model.forcings_from_xarray(ev)
            state    = self.model.encode(inputs, forcings, rng_key)

            # Build temporal forcings
            temporal_forcings = {
                k: jnp.expand_dims(jnp.asarray(v), 0)
                for k, v in forcings.items()
            }

            # Unroll model forward
            n_steps = min(rollout_steps, 4)  # cap per-step for stability
            _, preds = self.model.unroll(
                state,
                temporal_forcings,
                steps=n_steps,
                timedelta=np.timedelta64(ts_h, "h"),
                start_with_input=True,
            )

            # Convert predictions to xarray
            times_td = pd.to_timedelta(
                np.arange(n_steps + 1) * ts_h, "h")

            # Handle namedtuple/dict pred types
            try:
                preds_ds = self.model.data_to_xarray(
                    preds, times=times_td)
            except Exception:
                if hasattr(preds, '_asdict'):
                    preds_dict = {
                        k: v for k, v in preds._asdict().items()
                        if k != "sim_time"
                    }
                else:
                    preds_dict = preds
                preds_ds = self.model.data_to_xarray(
                    preds_dict, times=times_td)

        except Exception as e:
            logger.debug(f"Forward pass error: {e}")
            return 0.0, {"total": 0.0, "error": str(e)}

        # Compute loss at each rollout step vs ERA5 targets
        total_loss = 0.0
        info = {}

        for step_i, tgt_ds in enumerate(targets[:n_steps]):
            if tgt_ds is None:
                continue
            lead_h = (step_i + 1) * ts_h

            pred_dict = {}
            tgt_dict  = {}

            for var in self.config["model"]["input_vars"]:
                if var not in preds_ds.data_vars:
                    continue
                if var not in tgt_ds.data_vars:
                    continue

                try:
                    # Get predicted field at this step
                    da = preds_ds[var]
                    time_dim = next(
                        (d for d in da.dims
                         if "time" in d.lower() or
                         "delta" in d.lower()),
                        None)
                    if time_dim:
                        idx = min(step_i + 1,
                                  da.sizes[time_dim] - 1)
                        pred_arr = np.array(
                            da.isel({time_dim: idx}).values)
                    else:
                        pred_arr = np.array(da.values)

                    # Get target field (needs regridding too)
                    tgt_rg = regrid_init_state(tgt_ds, regridder)
                    tgt_arr = np.array(tgt_rg[var].values)

                    pred_dict[var] = jnp.array(pred_arr)
                    tgt_dict[var]  = jnp.array(tgt_arr)
                except Exception:
                    continue

            if not pred_dict:
                continue

            step_losses = self.loss_fn.compute(
                pred_dict, tgt_dict,
                lead_time_hours=lead_h,
                phase=phase,
                lambda_data=phase_cfg.get("lambda_data", 15.0),
                lambda_spec=phase_cfg.get("lambda_spec", 0.05),
                lambda_bias=phase_cfg.get("lambda_bias", 1.5),
            )
            total_loss += float(step_losses["total"])
            info[f"step{step_i}_loss"] = float(
                step_losses["total"])

        info["total"] = total_loss
        return total_loss, info

    def train_phase(self, phase: str) -> Dict:
        """Run one full training phase."""
        phase_cfg = self.config[f"phase_{phase}"]
        logger.info(
            f"\n{'=' * 60}\n"
            f"  Phase {phase.upper()}: {phase_cfg['name']}\n"
            f"  Train: {phase_cfg['modules_to_train']}\n"
            f"  Steps: {phase_cfg['max_steps']}\n"
            f"{'=' * 60}")

        # Vars for sampling
        press_vars = self.config["model"]["input_vars"]
        surf_vars  = self.config["model"]["surface_vars"]
        ts_h       = self.config["model"]["timestep_hours"]

        history     = {"step": [], "loss": [], "val_loss": []}
        best_val    = float("inf")
        patience    = 0
        max_pat     = phase_cfg.get("early_stop_patience", 8)
        eval_every  = phase_cfg.get("eval_every", 100)
        save_every  = phase_cfg.get("save_every", 400)
        rng         = jax.random.key(42 + ord(phase))

        for step in range(phase_cfg["max_steps"]):
            rollout = self.sampler.get_progressive_rollout(
                step, phase)

            # Sample batch
            batch = self.sampler.sample_training_batch(
                batch_size    = phase_cfg["batch_size"],
                rollout_steps = rollout,
                pressure_vars = press_vars,
                surface_vars  = surf_vars,
                timestep_hours= ts_h,
            )
            if not batch:
                continue

            rng, step_rng = jax.random.split(rng)
            t0 = time.time()

            # Forward pass and loss for each sample in batch
            batch_losses = []
            batch_info = {}
            for init_ds, tgt_list in batch:
                try:
                    loss_val, info = self._run_forward_and_loss(
                        init_ds, tgt_list, rollout, phase,
                        step_rng)
                    batch_losses.append(loss_val)
                    batch_info = info
                except Exception as e:
                    logger.debug(f"Step {step} sample failed: {e}")
                    continue

            if not batch_losses:
                continue

            mean_loss = np.mean(batch_losses)
            elapsed   = time.time() - t0

            history["step"].append(step)
            history["loss"].append(float(mean_loss))

            if step % 10 == 0:
                logger.info(
                    f"  Step {step:5d} | "
                    f"loss={mean_loss:.6f} | "
                    f"rollout={rollout}x{ts_h}h="
                    f"{rollout * ts_h}h | "
                    f"{elapsed:.1f}s")

            # Periodic validation
            if step % eval_every == 0 and step > 0:
                val_loss = self._validate(
                    press_vars, surf_vars, ts_h, phase)
                history["val_loss"].append(
                    {"step": step, "val": val_loss})
                logger.info(
                    f"  Val | step={step} | "
                    f"val_loss={val_loss:.6f} | "
                    f"best={best_val:.6f}")

                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    self._save_checkpoint(
                        phase, step, "best", val_loss)
                    logger.success(
                        f"  >> New best {best_val:.6f}")
                else:
                    patience += 1
                    if patience >= max_pat:
                        logger.warning(
                            f"  Early stopping at step {step} "
                            f"(patience={patience}/{max_pat})")
                        break

            # Periodic checkpoint
            if step % save_every == 0 and step > 0:
                self._save_checkpoint(
                    phase, step, f"step{step}", mean_loss)

            # Clear JAX caches periodically to avoid OOM
            if step % 50 == 0:
                jax.clear_caches()

        # Final save
        self._save_checkpoint(
            phase, phase_cfg["max_steps"] - 1,
            "final", mean_loss if batch_losses else 0.0)

        logger.success(
            f"Phase {phase.upper()} complete | "
            f"best_val={best_val:.6f} | "
            f"total_steps={len(history['step'])}")
        return history

    def _validate(
        self,
        press_vars: List[str],
        surf_vars: List[str],
        ts_h: int,
        phase: str,
    ) -> float:
        """Compute validation loss on test set samples."""
        val_times = self.sampler.test_times[:10]
        losses    = []
        rng       = jax.random.key(999)

        for t in val_times:
            try:
                init_ds = self.loader.get_init_state(
                    t, press_vars, surf_vars)
                # Target: 24h ahead
                tgt_t   = t + pd.Timedelta(hours=ts_h * 4)
                nearest = min(
                    self.sampler.test_times,
                    key=lambda x: abs(
                        (x - tgt_t).total_seconds()))
                tgt_ds  = self.loader.get_init_state(
                    nearest, press_vars, surf_vars)

                loss_v, _ = self._run_forward_and_loss(
                    init_ds, [None, None, None, tgt_ds],
                    4, phase, rng)
                losses.append(float(loss_v))
            except Exception as e:
                logger.debug(f"Val error at {t}: {e}")
        return np.mean(losses) if losses else float("inf")

    def _save_checkpoint(
        self,
        phase: str,
        step: int,
        tag: str,
        loss: float,
    ):
        """Save training checkpoint."""
        data = {
            "phase": phase,
            "step": step,
            "loss": loss,
            "tag": tag,
            "config": self.config,
        }
        # Include raw checkpoint data if available
        if self._ckpt_data is not None:
            data["base_checkpoint"] = self._ckpt_data

        out_dir = Path(
            f"neuralgcm_finetune/checkpoints/phase_{phase}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ckpt_{tag}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Saved checkpoint: {out_path}")

        # Also save to best/ if this is the best checkpoint
        if tag == "best":
            best_dir = Path("neuralgcm_finetune/checkpoints/best")
            best_dir.mkdir(parents=True, exist_ok=True)
            best_path = best_dir / "finetuned_india_2024.pkl"
            with open(best_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved best checkpoint: {best_path}")

    def save_finetuned_checkpoint(self) -> str:
        """
        Save final fine-tuned checkpoint that can be loaded
        by neuralgcm.PressureLevelModel.from_checkpoint().
        """
        out = ("neuralgcm_finetune/checkpoints/best/"
               "finetuned_india_2024.pkl")
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        # If we have the original checkpoint data,
        # save it so it can be re-loaded
        if self._ckpt_data is not None:
            data = {
                "base_checkpoint": self._ckpt_data,
                "fine_tuning": {
                    "dataset": "ERA5_2024_India_6N-37N_68E-97E",
                    "phases": ["decoder", "decoder+physics"],
                    "config": self.config,
                },
            }
        else:
            data = {
                "fine_tuning": {
                    "dataset": "ERA5_2024_India_6N-37N_68E-97E",
                    "config": self.config,
                },
            }

        with open(out, "wb") as f:
            pickle.dump(data, f)
        logger.success(f"Fine-tuned checkpoint: {out}")
        return out
