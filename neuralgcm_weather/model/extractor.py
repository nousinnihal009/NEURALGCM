"""
Variable Extractor
==================
Extracts time series for all weather variables at a specific
lat/lon grid point from NeuralGCM forecast output.
All physical unit conversions are correct and documented.
"""

import numpy as np
import xarray as xr
from loguru import logger
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class ForecastPoint:
    """All extracted weather variables for one location."""
    location_name: str
    lat:           float
    lon:           float
    model_lat:     float
    model_lon:     float
    dates:         list
    days:          int

    # Thermodynamic
    temperature_c_850:   Optional[np.ndarray] = None
    temperature_c_500:   Optional[np.ndarray] = None
    rh_850:              Optional[np.ndarray] = None
    rh_500:              Optional[np.ndarray] = None
    specific_humidity_850: Optional[np.ndarray] = None
    tpw_mm:              Optional[np.ndarray] = None  # FIXED: no *1000

    # Wind
    u_850: Optional[np.ndarray] = None
    v_850: Optional[np.ndarray] = None
    u_500: Optional[np.ndarray] = None
    v_500: Optional[np.ndarray] = None
    u_250: Optional[np.ndarray] = None
    v_250: Optional[np.ndarray] = None
    wind_speed_850: Optional[np.ndarray] = None
    wind_speed_500: Optional[np.ndarray] = None
    wind_speed_250: Optional[np.ndarray] = None
    wind_dir_850:   Optional[np.ndarray] = None

    # Pressure / height
    z500_m:    Optional[np.ndarray] = None
    z850_m:    Optional[np.ndarray] = None
    mslp_hpa:  Optional[np.ndarray] = None   # FIXED: validated range

    # Stability
    lapse_rate: Optional[np.ndarray] = None

    # Cloud
    clwc_gkg_850: Optional[np.ndarray] = None
    ciwc_gkg_850: Optional[np.ndarray] = None

    # Dynamics
    vorticity_850:  Optional[np.ndarray] = None
    divergence_850: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict."""
        import pandas as pd
        out = {
            "location": self.location_name,
            "lat": self.lat, "lon": self.lon,
            "model_lat": self.model_lat,
            "model_lon": self.model_lon,
            "dates": [d.strftime("%Y-%m-%d") for d in self.dates],
        }
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                out[k] = [round(float(x), 4) if not np.isnan(x)
                          else None for x in v]
        return out


class VariableExtractor:
    """
    Extracts all weather variables at a single lat/lon point
    from a NeuralGCM forecast xarray Dataset.
    """

    def __init__(self, ds: xr.Dataset, n_steps: int):
        self.ds = ds
        self.n  = n_steps
        self._detect_dims()
        logger.info(
            f"Extractor ready | "
            f"dims: time={self.TIME_DIM} "
            f"lat={self.LAT_DIM} lon={self.LON_DIM}")

    def _detect_dims(self):
        """Detect actual dimension names in NeuralGCM output."""
        self.LAT_DIM = self.LON_DIM = self.TIME_DIM = None
        for da in self.ds.data_vars.values():
            for d in da.dims:
                dl = d.lower()
                if "lat" in dl and self.LAT_DIM is None:
                    self.LAT_DIM = d
                if "lon" in dl and self.LON_DIM is None:
                    self.LON_DIM = d
                if (("time" in dl or "delta" in dl or "step" in dl)
                        and self.TIME_DIM is None):
                    self.TIME_DIM = d
            if self.LAT_DIM and self.LON_DIM and self.TIME_DIM:
                break
        # Fallback to known NeuralGCM dim names
        self.LAT_DIM  = self.LAT_DIM  or "latitude"
        self.LON_DIM  = self.LON_DIM  or "longitude"
        self.TIME_DIM = self.TIME_DIM or "prediction_timedelta"

    def _get_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lat/lon arrays of NeuralGCM model grid."""
        for da in self.ds.data_vars.values():
            if (self.LAT_DIM in da.coords and
                    self.LON_DIM in da.coords):
                lats = np.array(da.coords[self.LAT_DIM])
                lons = np.array(da.coords[self.LON_DIM]) % 360
                return lats, lons
        # Gaussian T63 fallback
        return (np.linspace(87.863, -87.863, 64),
                np.linspace(0, 357.1875, 128))

    def find_grid_point(self, lat: float, lon: float
                        ) -> Tuple[int, int, float, float]:
        """Find nearest NeuralGCM grid point to (lat, lon)."""
        grid_lats, grid_lons = self._get_grid()
        lon_360  = lon % 360
        lat_idx  = int(np.argmin(np.abs(grid_lats - lat)))
        lon_idx  = int(np.argmin(np.abs(grid_lons - lon_360)))
        return (lat_idx, lon_idx,
                float(grid_lats[lat_idx]),
                float(grid_lons[lon_idx]))

    def extract(self, var: str, level: Optional[int] = None,
                lat_idx: int = 0, lon_idx: int = 0) -> Optional[np.ndarray]:
        """Extract time series at grid point for one variable."""
        if var not in self.ds.data_vars:
            return None
        da   = self.ds[var]
        dims = list(da.dims)
        idx  = {self.TIME_DIM: slice(0, self.n)}
        if self.LAT_DIM in dims:
            idx[self.LAT_DIM] = lat_idx
        if self.LON_DIM in dims:
            idx[self.LON_DIM] = lon_idx
        da = da.isel(**idx)
        if level is not None and "level" in da.dims:
            da = da.sel(level=level, method="nearest")
        arr = np.array(da).flatten()
        result = arr[:self.n]
        if len(result) < self.n:
            result = np.pad(result,
                            (0, self.n - len(result)),
                            constant_values=np.nan)
        return result

    @staticmethod
    def q_to_rh(q: np.ndarray, T_K: np.ndarray,
                p_hPa: float = 850) -> np.ndarray:
        """
        Specific humidity (kg/kg) -> Relative Humidity (%).
        Tetens formula for saturation vapour pressure.
        """
        T_C = T_K - 273.15
        es  = 6.1078 * np.exp(17.27 * T_C / (T_C + 237.3))
        qs  = 0.622 * es / (p_hPa - es)
        return np.clip(q / qs * 100, 0, 100)

    @staticmethod
    def wind_direction(u: np.ndarray,
                       v: np.ndarray) -> np.ndarray:
        """Meteorological wind direction 0deg=N, 90deg=E, 180deg=S, 270deg=W."""
        return (270 - np.rad2deg(np.arctan2(v, u))) % 360

    @staticmethod
    def lapse_rate(T_low_K: np.ndarray, T_high_K: np.ndarray,
                   p_low: float = 850,
                   p_high: float = 500) -> np.ndarray:
        """Lapse rate between two pressure levels (deg C/km)."""
        T_mean = (T_low_K + T_high_K) / 2
        dz = (287.05 * T_mean / 9.80665) * np.log(p_low / p_high)
        return (T_low_K - T_high_K) / dz * 1000

    def compute_tpw(self, lat_idx: int,
                    lon_idx: int) -> Optional[np.ndarray]:
        """
        Total Precipitable Water (mm) — vertical integral of q.
        FIXED: result is in mm (kg/m2), no extra *1000 conversion.
        """
        levels = [1000, 925, 850, 700, 600, 500,
                  400, 300, 250, 200, 150, 100]
        q_cols = []
        for lev in levels:
            q = self.extract("specific_humidity", lev,
                             lat_idx, lon_idx)
            if q is not None:
                q_cols.append((lev, q))
        if len(q_cols) < 2:
            return None
        pw = np.zeros(self.n)
        for i in range(len(q_cols) - 1):
            p1, q1 = q_cols[i]
            p2, q2 = q_cols[i + 1]
            # Trapezoidal: PW = integral(q * dp / g)
            # dp in Pa, q in kg/kg -> result in kg/m2 = mm
            dp   = abs(p2 - p1) * 100   # hPa -> Pa
            pw  += (q1 + q2) / 2 * dp / 9.80665
        # pw is now in kg/m2 = mm of liquid water — NO *1000
        return pw

    def extract_all(self, location_name: str,
                    lat: float, lon: float,
                    forecast_dates: list) -> ForecastPoint:
        """Extract all weather variables for a location."""
        lat_idx, lon_idx, model_lat, model_lon = \
            self.find_grid_point(lat, lon)
        logger.info(
            f"Extracting {location_name} | "
            f"requested=({lat:.4f},{lon:.4f}) | "
            f"model=({model_lat:.2f},{model_lon:.2f}) | "
            f"idx=({lat_idx},{lon_idx})")

        ex = lambda v, l=None: self.extract(
            v, l, lat_idx, lon_idx)

        # Temperature
        T_K_850 = ex("temperature", 850)
        T_K_500 = ex("temperature", 500)
        T_C_850 = T_K_850 - 273.15 if T_K_850 is not None else None
        T_C_500 = T_K_500 - 273.15 if T_K_500 is not None else None

        # Humidity
        Q_850  = ex("specific_humidity", 850)
        Q_500  = ex("specific_humidity", 500)
        RH_850 = (self.q_to_rh(Q_850, T_K_850, 850)
                  if Q_850 is not None and T_K_850 is not None
                  else None)
        RH_500 = (self.q_to_rh(Q_500, T_K_500, 500)
                  if Q_500 is not None and T_K_500 is not None
                  else None)

        # Wind
        U_850 = ex("u_component_of_wind", 850)
        V_850 = ex("v_component_of_wind", 850)
        U_500 = ex("u_component_of_wind", 500)
        V_500 = ex("v_component_of_wind", 500)
        U_250 = ex("u_component_of_wind", 250)
        V_250 = ex("v_component_of_wind", 250)

        def ws(u, v):
            return (np.sqrt(u**2 + v**2)
                    if u is not None and v is not None else None)

        # Geopotential -> height (m)
        Z_500  = ex("geopotential", 500)
        Z_850  = ex("geopotential", 850)
        Z500_m = Z_500 / 9.80665 if Z_500 is not None else None
        Z850_m = Z_850 / 9.80665 if Z_850 is not None else None

        # Surface pressure — FIXED validation
        LOG_PS = ex("log_surface_pressure")
        SP_hPa = None
        if LOG_PS is not None:
            sp_cand = np.exp(LOG_PS) / 100.0
            if 800 < np.nanmean(sp_cand) < 1100:
                SP_hPa = sp_cand
            else:
                logger.warning(
                    f"SP sanity failed: mean={np.nanmean(sp_cand):.1f}hPa")

        # Lapse rate
        LR = (self.lapse_rate(T_K_850, T_K_500)
              if T_K_850 is not None and T_K_500 is not None
              else None)

        # Cloud water (g/kg)
        CLWC = ex("specific_cloud_liquid_water_content", 850)
        CIWC = ex("specific_cloud_ice_water_content", 850)

        # TPW — FIXED: result in mm, no *1000
        TPW = self.compute_tpw(lat_idx, lon_idx)

        # Log what we got
        got = [k for k, v in [
            ("T850",T_C_850),("T500",T_C_500),
            ("RH850",RH_850),("TPW",TPW),
            ("WS850",ws(U_850,V_850)),
            ("Z500",Z500_m),("SP",SP_hPa),
            ("LR",LR),("CLWC",CLWC)
        ] if v is not None]
        logger.success(f"Extracted variables: {got}")

        return ForecastPoint(
            location_name = location_name,
            lat=lat, lon=lon,
            model_lat=model_lat, model_lon=model_lon,
            dates=forecast_dates,
            days=self.n - 1,
            temperature_c_850 = T_C_850,
            temperature_c_500 = T_C_500,
            rh_850            = RH_850,
            rh_500            = RH_500,
            specific_humidity_850 = Q_850,
            tpw_mm            = TPW,
            u_850=U_850, v_850=V_850,
            u_500=U_500, v_500=V_500,
            u_250=U_250, v_250=V_250,
            wind_speed_850    = ws(U_850, V_850),
            wind_speed_500    = ws(U_500, V_500),
            wind_speed_250    = ws(U_250, V_250),
            wind_dir_850      = (self.wind_direction(U_850, V_850)
                                 if U_850 is not None else None),
            z500_m            = Z500_m,
            z850_m            = Z850_m,
            mslp_hpa          = SP_hPa,
            lapse_rate        = LR,
            clwc_gkg_850      = CLWC * 1000 if CLWC is not None else None,
            ciwc_gkg_850      = CIWC * 1000 if CIWC is not None else None,
            vorticity_850     = ex("vorticity", 850),
            divergence_850    = ex("divergence", 850),
        )
