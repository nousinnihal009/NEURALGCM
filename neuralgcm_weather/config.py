import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config(path: str = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = load_config()

class ModelConfig:
    checkpoint:     str   = _cfg["model"]["checkpoint"]
    gcs_bucket:     str   = _cfg["model"]["gcs_bucket"]
    forecast_days:  int   = _cfg["model"]["forecast_days"]
    timestep_hours: int   = _cfg["model"]["timestep_hours"]

class DataConfig:
    mode:           str   = _cfg["data"]["mode"]
    era5_zarr:      str   = _cfg["data"]["era5_zarr"]
    cache_dir:      str   = _cfg["data"]["cache_dir"]
    ecmwf_lag_hours:int   = _cfg["data"]["ecmwf_lag_hours"]

class LocationConfig:
    name: str = _cfg["location"]["name"]
    lat:  float = float(_cfg["location"]["lat"])
    lon:  float = float(_cfg["location"]["lon"])

LOCATION = LocationConfig()

class OutputConfig:
    save_dir:  str  = _cfg["output"]["save_dir"]
    save_png:  bool = _cfg["output"]["save_png"]
    save_json: bool = _cfg["output"]["save_json"]
    save_csv:  bool = _cfg["output"]["save_csv"]
    png_dpi:   int  = _cfg["output"]["png_dpi"]

class SanityConfig:
    bounds = _cfg["sanity"]
    def check(self, var: str, values) -> list:
        import numpy as np
        if var not in self.bounds:
            return []
        lo = self.bounds[var]["min"]
        hi = self.bounds[var]["max"]
        violations = []
        for i, v in enumerate(values):
            if not np.isnan(v) and (v < lo or v > hi):
                violations.append(
                    f"Day {i}: {var}={v:.2f} outside [{lo}, {hi}]")
        return violations

MODEL   = ModelConfig()
DATA    = DataConfig()
OUTPUT  = OutputConfig()
SANITY  = SanityConfig()
CFG_RAW = _cfg
