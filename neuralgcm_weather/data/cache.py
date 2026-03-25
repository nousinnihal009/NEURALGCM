"""
Disk Cache Manager
==================
Simple disk caching for downloaded data and regridded states.
"""

import os
import hashlib
import pickle
from pathlib import Path
from loguru import logger


class DiskCache:
    """Simple file-based cache for intermediate data."""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.pkl"

    def get(self, key: str):
        path = self._key_path(key)
        if path.exists():
            logger.debug(f"Cache hit: {key}")
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def put(self, key: str, value):
        path = self._key_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)
        logger.debug(f"Cached: {key} -> {path}")

    def clear(self):
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        logger.info("Cache cleared")
