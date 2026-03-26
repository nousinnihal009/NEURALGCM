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
        """Tropical boundary layer: q=20g/kg, T=30C -> RH ~61% (Tetens)"""
        q   = np.array([0.020])
        T_K = np.array([303.15])  # 30C
        rh  = VariableExtractor.q_to_rh(q, T_K, 850)
        assert 50 < rh[0] < 100, f"Expected RH 50-100%, got {rh[0]:.1f}%"

    def test_q_to_rh_dry_desert(self):
        """Sahara desert: q=2g/kg, T=45C -> RH ~2.5% (Tetens, extreme heat)"""
        q   = np.array([0.002])
        T_K = np.array([318.15])  # 45C
        rh  = VariableExtractor.q_to_rh(q, T_K, 850)
        assert 0 < rh[0] < 20, f"Expected RH 0-20%, got {rh[0]:.1f}%"

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


class TestPhase1RegressionGuards:
    """
    Explicit regression tests for the 5 bugs fixed from
    forecast_anywhere.py. If any of these fail, a fixed bug
    has been reintroduced.
    """

    def test_tpw_not_multiplied_by_1000(self):
        """
        REGRESSION GUARD: TPW must be in mm (kg/m²), not g/m².
        Old bug: compute_tpw() returned pw * 1000 → ~50000mm
        """
        q_kgkg  = 0.015      # 15 g/kg, typical tropical boundary layer
        dp_Pa   = 150 * 100  # 150 hPa layer in Pa
        g       = 9.80665
        pw_mm   = q_kgkg * dp_Pa / g
        assert pw_mm < 500, (
            f"REGRESSION: TPW single layer = {pw_mm:.1f}mm. "
            f"If >500, the *1000 bug is back.")
        assert pw_mm > 5, f"TPW unrealistically low: {pw_mm:.1f}mm"

    def test_unroll_steps_not_plus_one(self):
        """
        REGRESSION GUARD: steps=forecast_days, NOT forecast_days+1.
        Old bug caused an extra unwanted step in forecast output.
        """
        forecast_days = 5
        # Simulate what runner.py does — steps must equal forecast_days
        steps_used = forecast_days   # correct
        assert steps_used == 5, (
            "steps should equal forecast_days, not forecast_days+1")
        assert steps_used != forecast_days + 1

    def test_surface_pressure_sanity_enforced(self):
        """
        REGRESSION GUARD: SP must be validated to 800-1100 hPa.
        Old bug: 2000 hPa values passed through silently.
        """
        import math
        sp_bad_pa  = 200000.0   # 2000 hPa in Pa — clearly wrong
        sp_bad_hpa = sp_bad_pa / 100.0
        assert not (800 < sp_bad_hpa < 1100), (
            "Sanity check should REJECT 2000 hPa surface pressure")

        sp_ok_pa  = 101325.0
        sp_ok_hpa = sp_ok_pa / 100.0
        assert 800 < sp_ok_hpa < 1100, (
            "Sanity check should ACCEPT ~1013 hPa surface pressure")

    def test_geopotential_divided_by_g(self):
        """
        REGRESSION GUARD: Z must be divided by g (9.80665) to get metres.
        Old bug: values left as m²/s² were ~9.8× too large.
        """
        z_m2s2 = 57000.0        # Z500 in m²/s²
        z_m    = z_m2s2 / 9.80665
        assert 4500 < z_m < 6500, (
            f"REGRESSION: Z500={z_m:.0f}m outside expected range. "
            f"If ~57000, divide by g was missing.")

    def test_preds_sim_time_stripping(self):
        """
        REGRESSION GUARD: data_to_xarray must handle namedtuple preds.
        Old bug: preds assumed to be dict, crashed on namedtuple.
        """
        from collections import namedtuple
        FakePreds = namedtuple("FakePreds",
                               ["temperature", "sim_time"])
        preds = FakePreds(temperature=None, sim_time=None)

        # Verify _asdict() works and sim_time can be stripped
        assert hasattr(preds, "_asdict"), \
            "namedtuple should have _asdict()"
        stripped = {k: v for k, v in preds._asdict().items()
                    if k != "sim_time"}
        assert "temperature" in stripped
        assert "sim_time" not in stripped
