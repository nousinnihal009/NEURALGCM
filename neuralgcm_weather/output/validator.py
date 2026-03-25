"""
Physical Sanity Validator
==========================
Checks all forecast output against physically plausible bounds.
Raises warnings (not errors) on violations so pipeline continues.
"""

import numpy as np
from loguru import logger
from typing import List, Tuple


BOUNDS = {
    "temperature_c_850":  (-90,   60,  "°C"),
    "temperature_c_500":  (-90,   10,  "°C"),
    "rh_850":             (0,    100,  "%"),
    "rh_500":             (0,    100,  "%"),
    "tpw_mm":             (0,    100,  "mm"),
    "wind_speed_850":     (0,    120,  "m/s"),
    "wind_speed_500":     (0,    150,  "m/s"),
    "wind_speed_250":     (0,    250,  "m/s"),
    "z500_m":             (4500, 6500, "m"),
    "mslp_hpa":           (870,  1085, "hPa"),
    "lapse_rate":         (-5,   20,   "°C/km"),
}


def validate_forecast(fp) -> Tuple[bool, List[str]]:
    """
    Validate a ForecastPoint object against physical bounds.
    Returns (all_ok, list_of_violation_messages).
    """
    violations = []
    for var, (lo, hi, unit) in BOUNDS.items():
        arr = getattr(fp, var, None)
        if arr is None:
            continue
        for i, v in enumerate(arr):
            if np.isnan(v):
                continue
            if v < lo or v > hi:
                msg = (f"Day {i} | {var}={v:.2f}{unit} "
                       f"outside [{lo},{hi}]")
                violations.append(msg)
                logger.warning(f"SANITY FAIL: {msg}")

    if not violations:
        logger.success(
            f"All variables passed sanity checks for "
            f"{fp.location_name}")

    return len(violations) == 0, violations
