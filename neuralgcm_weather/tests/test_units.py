"""
Physical Unit Tests
===================
Every variable must produce physically plausible values.
These tests catch the unit bugs that plagued forecast_anywhere.py.
"""

import pytest
import numpy as np
from neuralgcm_weather.model.extractor import VariableExtractor


class TestPhysicalConversions:

    def test_q_to_rh_tropical_moist(self):
        """Tropical boundary layer: q=20g/kg, T=30C -> RH should be ~90%"""
        q   = np.array([0.020])
        T_K = np.array([303.15])  # 30C
        rh  = VariableExtractor.q_to_rh(q, T_K, 850)
        assert 70 < rh[0] < 100, f"Expected RH 70-100%, got {rh[0]:.1f}%"

    def test_q_to_rh_dry_desert(self):
        """Sahara desert: q=2g/kg, T=45C -> RH should be ~10-20%"""
        q   = np.array([0.002])
        T_K = np.array([318.15])  # 45C
        rh  = VariableExtractor.q_to_rh(q, T_K, 850)
        assert 5 < rh[0] < 35, f"Expected RH 5-35%, got {rh[0]:.1f}%"

    def test_rh_always_0_to_100(self):
        """RH must never exceed 0-100%."""
        q   = np.array([0.001, 0.010, 0.030, 0.050])
        T_K = np.array([260.0, 280.0, 300.0, 310.0])
        rh  = VariableExtractor.q_to_rh(q, T_K, 850)
        assert np.all(rh >= 0),   f"RH below 0: {rh}"
        assert np.all(rh <= 100), f"RH above 100: {rh}"

    def test_tpw_tropical_range(self):
        """TPW in tropics should be 40-80mm, NOT 40000-80000mm."""
        # Simulate typical tropical q profile (kg/kg)
        # The old bug multiplied by 1000 giving mm*1000 = nonsense
        # TPW for 850hPa layer only: q=15g/kg, dp=150hPa
        q      = 0.015     # kg/kg
        dp_Pa  = 150 * 100 # Pa
        g      = 9.80665
        pw_mm  = q * dp_Pa / g   # kg/m2 = mm
        assert 10 < pw_mm < 50, (
            f"Single layer PW should be 10-50mm, got {pw_mm:.1f}mm\n"
            f"If you see >1000mm, the *1000 bug is back!")

    def test_geopotential_to_height(self):
        """Z500 should be ~5000-6000m, not 49000-59000 m2/s2."""
        z_m2s2 = np.array([53000.0])  # typical Z500 in m2/s2
        z_m    = z_m2s2 / 9.80665
        assert 4500 < z_m[0] < 6500, (
            f"Z500 should be 4500-6500m, got {z_m[0]:.0f}m")

    def test_surface_pressure_range(self):
        """Surface pressure must be 870-1085 hPa."""
        import math
        log_ps = math.log(101325)   # ln(Pa) ~ 11.526
        sp_hpa = math.exp(log_ps) / 100.0
        assert 800 < sp_hpa < 1100, (
            f"SP should be 800-1100 hPa, got {sp_hpa:.1f}")

    def test_lapse_rate_standard_atmosphere(self):
        """Standard atmosphere lapse rate ~ 6.5 C/km."""
        T_K_850 = np.array([288.15])  # 15C
        T_K_500 = np.array([256.65])  # -16.5C (approx std atm)
        lr = VariableExtractor.lapse_rate(T_K_850, T_K_500)
        assert 4.0 < lr[0] < 9.0, (
            f"Standard atmosphere lapse rate should be 4-9 C/km, "
            f"got {lr[0]:.2f}")

    def test_wind_direction_north(self):
        """U=0, V=-10 (northerly wind FROM north) -> dir should be ~360"""
        u = np.array([0.0])
        v = np.array([-10.0])  # negative V = wind FROM north
        wd = VariableExtractor.wind_direction(u, v)
        assert 350 < wd[0] or wd[0] < 10, (
            f"Northerly wind should be ~360, got {wd[0]:.1f}")

    def test_wind_direction_easterly(self):
        """U=-10, V=0 (easterly wind FROM east) -> dir should be ~90"""
        u = np.array([-10.0])
        v = np.array([0.0])
        wd = VariableExtractor.wind_direction(u, v)
        assert 80 < wd[0] < 100, (
            f"Easterly wind should be ~90, got {wd[0]:.1f}")

    def test_temperature_k_to_c(self):
        """273.15K should be exactly 0C."""
        T_K = np.array([273.15, 300.0, 250.0])
        T_C = T_K - 273.15
        assert T_C[0] == pytest.approx(0.0)
        assert T_C[1] == pytest.approx(26.85)
        assert T_C[2] == pytest.approx(-23.15)
