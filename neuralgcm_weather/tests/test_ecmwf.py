"""
ECMWF Loader Tests
==================
Tests for the ECMWF open data download and conversion functions.
These tests are offline-safe (mock network calls).
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from neuralgcm_weather.data.ecmwf_loader import (
    get_latest_available_date,
    ECMWF_VARS,
    ECMWF_TO_ERA5,
    PRESSURE_LEVELS,
)


class TestECMWFLoader:

    def test_latest_date_returns_timestamp(self):
        """get_latest_available_date should return a Timestamp."""
        dt = get_latest_available_date(lag_hours=6)
        assert isinstance(dt, pd.Timestamp)

    def test_latest_date_is_6h_aligned(self):
        """Returned date should be aligned to 0/6/12/18 UTC."""
        dt = get_latest_available_date(lag_hours=6)
        assert dt.hour in [0, 6, 12, 18], (
            f"Hour should be 0/6/12/18, got {dt.hour}")

    def test_latest_date_is_in_past(self):
        """Returned date should be in the past."""
        dt = get_latest_available_date(lag_hours=6)
        now = pd.Timestamp.utcnow()
        assert dt <= now, f"Latest date {dt} should be <= now {now}"

    def test_ecmwf_vars_list(self):
        """ECMWF_VARS should contain the expected variables."""
        assert "u" in ECMWF_VARS
        assert "t" in ECMWF_VARS
        assert "q" in ECMWF_VARS
        assert "z" in ECMWF_VARS

    def test_ecmwf_to_era5_mapping(self):
        """Variable renaming map should have correct mappings."""
        assert ECMWF_TO_ERA5["t"] == "temperature"
        assert ECMWF_TO_ERA5["u"] == "u_component_of_wind"
        assert ECMWF_TO_ERA5["q"] == "specific_humidity"

    def test_pressure_levels_count(self):
        """Should have standard 22 pressure levels."""
        assert len(PRESSURE_LEVELS) == 22
        assert 1000 in PRESSURE_LEVELS
        assert 500 in PRESSURE_LEVELS
        assert 850 in PRESSURE_LEVELS
        assert 1 in PRESSURE_LEVELS

    def test_pressure_levels_sorted(self):
        """Pressure levels should be sorted ascending."""
        assert PRESSURE_LEVELS == sorted(PRESSURE_LEVELS)
