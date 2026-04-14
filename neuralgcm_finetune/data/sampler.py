"""
Training Batch Sampler
=======================
Samples (init_time, target_times) pairs from the merged dataset.
Implements train/test chronological split with no data leakage.
Handles the Indian subcontinent regional dataset (1 year, 2024).
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Tuple, Optional
from loguru import logger


class IndiaERA5Sampler:
    """
    Samples training batches from the 2024 Indian subcontinent
    ERA5 dataset.

    Train set: Jan 1 to Dec 1, 2024 (first ~335 days)
    Test set:  Dec 1 to Dec 31, 2024 (last 30 days — held out)

    This split ensures:
    - Test set is strictly in the future vs training data
    - No data leakage
    - Monsoon season (Jun-Sep) is in the training set
    - Post-monsoon withdrawal (Oct-Nov) is in training
    - Winter onset (Dec) is the test set
    """

    def __init__(
        self,
        loader,                    # ERA5GRIBLoader instance
        config: dict,
        rng_seed: int = 42,
    ):
        self.loader  = loader
        self.config  = config
        self.rng     = np.random.default_rng(rng_seed)
        self._all_times: List[pd.Timestamp] = []
        self._train_times: List[pd.Timestamp] = []
        self._test_times: List[pd.Timestamp] = []
        self._setup()

    def _setup(self):
        """Build train/test time lists."""
        all_times = self.loader.get_available_times()
        if not all_times:
            raise RuntimeError(
                "No time steps found in dataset")

        test_days  = self.config["data"].get("test_days", 30)
        split_time = (all_times[-1]
                      - pd.Timedelta(days=test_days))

        self._all_times   = all_times
        self._train_times = [t for t in all_times
                             if t < split_time]
        self._test_times  = [t for t in all_times
                             if t >= split_time]

        logger.info(
            f"Sampler ready:\n"
            f"  Total steps:  {len(all_times)}\n"
            f"  Train steps:  {len(self._train_times)} "
            f"({all_times[0].strftime('%Y-%m-%d')} to "
            f"{self._train_times[-1].strftime('%Y-%m-%d')})\n"
            f"  Test steps:   {len(self._test_times)} "
            f"({self._test_times[0].strftime('%Y-%m-%d')} to "
            f"{all_times[-1].strftime('%Y-%m-%d')})")

    @property
    def train_times(self) -> List[pd.Timestamp]:
        return self._train_times

    @property
    def test_times(self) -> List[pd.Timestamp]:
        return self._test_times

    def sample_training_batch(
        self,
        batch_size: int,
        rollout_steps: int,
        pressure_vars: List[str],
        surface_vars: List[str],
        timestep_hours: int = 6,
    ) -> List[Tuple[xr.Dataset, List[Optional[xr.Dataset]]]]:
        """
        Sample a batch of (init, [target_t1, ..., target_tN]) tuples.

        rollout_steps: number of forward steps (e.g. 4 = 24h at 6h step)
        Each target is one timestep ahead of the previous.
        """
        # Valid init times: must have rollout_steps future steps
        max_t = self._train_times[-1]
        rollout_td = pd.Timedelta(hours=rollout_steps * timestep_hours)
        valid = [t for t in self._train_times
                 if t + rollout_td <= max_t]

        if len(valid) < batch_size:
            logger.warning(
                f"Only {len(valid)} valid init times for "
                f"{rollout_steps}-step rollout. "
                f"Reducing batch to {max(1, len(valid))}")
            batch_size = max(1, len(valid))

        chosen_idx = self.rng.choice(
            len(valid), size=batch_size, replace=False)
        batch = []

        for idx in chosen_idx:
            init_t  = valid[idx]
            init_ds = self.loader.get_init_state(
                init_t, pressure_vars, surface_vars)
            if init_ds is None:
                continue

            targets = []
            for step in range(1, rollout_steps + 1):
                tgt_t    = init_t + pd.Timedelta(
                    hours=step * timestep_hours)
                nearest  = min(
                    self._train_times,
                    key=lambda t: abs((t - tgt_t).total_seconds()))
                gap_h    = abs(
                    (nearest - tgt_t).total_seconds()) / 3600
                if gap_h <= timestep_hours / 2:
                    tgt_ds = self.loader.get_init_state(
                        nearest, pressure_vars, surface_vars)
                    targets.append(tgt_ds)
                else:
                    targets.append(None)

            if any(t is not None for t in targets):
                batch.append((init_ds, targets))

        return batch

    def get_progressive_rollout(
        self, step: int, phase: str
    ) -> int:
        """
        Progressive rollout schedule tailored for India 2024 dataset.
        Fewer steps than global fine-tuning (smaller dataset).

        Phase A (decoder): alternates 1 step (6h) and 4 steps (24h)
        Phase B (physics): steps up 4->8 based on training progress
        """
        if phase == "a":
            # Simple alternation for decoder fine-tuning
            return 4 if step % 3 == 0 else 1
        else:
            # Progressive rollout for physics fine-tuning
            # Start short, extend as training stabilises
            if step < 300:
                return 4    # 24h
            elif step < 1000:
                return 6    # 36h
            else:
                return 8    # 48h
