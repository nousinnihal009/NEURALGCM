"""
NeuralGCM Checkpoint Loader
============================
Loads and caches the NeuralGCM pretrained model checkpoint.
Singleton pattern — checkpoint loaded once per process.
"""

import os
import time
import pickle
from pathlib import Path
from loguru import logger
from typing import Optional

_MODEL_CACHE = {}   # module-level singleton


def load_checkpoint(
    model_name: str = "v1/deterministic_2_8_deg.pkl",
    gcs_bucket: str = "gs://neuralgcm/models",
    local_cache_dir: str = "./cache/checkpoints",
    max_retries: int = 3,
):
    """
    Load NeuralGCM checkpoint with caching.
    First call downloads from GCS (~200MB).
    Subsequent calls return cached model in memory.
    """
    global _MODEL_CACHE
    cache_key = model_name

    # Return cached if already loaded
    if cache_key in _MODEL_CACHE:
        logger.debug(f"Checkpoint in memory: {model_name}")
        return _MODEL_CACHE[cache_key]

    # Check local disk cache
    Path(local_cache_dir).mkdir(parents=True, exist_ok=True)
    local_path = Path(local_cache_dir) / model_name.replace("/", "_")
    if local_path.exists():
        logger.info(f"Loading checkpoint from disk: {local_path}")
        with open(local_path, "rb") as f:
            ckpt = pickle.load(f)
        model = _build_model(ckpt, model_name)
        _MODEL_CACHE[cache_key] = model
        return model

    # Download from GCS
    import gcsfs
    gcs = gcsfs.GCSFileSystem(token="anon")
    gcs_path = f"{gcs_bucket}/{model_name}"

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Downloading checkpoint from GCS "
                f"(attempt {attempt}/{max_retries}): {gcs_path}")
            with gcs.open(gcs_path, "rb") as f:
                ckpt = pickle.load(f)

            # Save to local disk for next time
            with open(local_path, "wb") as f:
                pickle.dump(ckpt, f)
            logger.success(
                f"Checkpoint saved to disk: {local_path}")

            model = _build_model(ckpt, model_name)
            _MODEL_CACHE[cache_key] = model
            return model

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)

    raise RuntimeError(
        f"Failed to load checkpoint after {max_retries} attempts: "
        f"{gcs_path}")


def _build_model(ckpt, model_name: str):
    """Build NeuralGCM model from checkpoint."""
    import neuralgcm
    logger.info("Building NeuralGCM model from checkpoint...")
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    logger.success(
        f"Model ready | "
        f"checkpoint={model_name} | "
        f"input_vars={model.input_variables} | "
        f"forcing_vars={model.forcing_variables}")
    return model


def clear_cache():
    """Clear in-memory model cache (for testing)."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
