"""
ERA5 GRIB Loader for NeuralGCM Fine-Tuning
============================================
Handles the exact 7-file structure:
  Single-level:   data1.grib  (Jan-Mar 2024)
                  data2.grib  (Apr-Jun 2024)
                  data3.grib  (Jul-Sep 2024)
                  data4.grib  (Oct-Dec 2024)
  Pressure-level: pressurelevel1.grib  (50-250 hPa)
                  pressurelevel2.grib  (350-750 hPa)
                  pressurelevel3.grib  (800-1000 hPa)

All 7 files cover:
  Region: 6°N-37°N, 68°E-97°E (Indian subcontinent)
  Times:  00:00, 06:00, 12:00, 18:00 UTC
  Year:   2024
"""

import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import xarray as xr
import pandas as pd
from loguru import logger

# ── Variable rename maps ──────────────────────────────────────
# GRIB shortName → NeuralGCM name
GRIB_SHORT_TO_NEURALGCM = {
    # Pressure-level variables (in your pressurelevel*.grib files)
    "u":    "u_component_of_wind",
    "v":    "v_component_of_wind",
    "t":    "temperature",
    "q":    "specific_humidity",
    "z":    "geopotential",
    "clwc": "specific_cloud_liquid_water_content",
    "ciwc": "specific_cloud_ice_water_content",
    "w":    "vertical_velocity",
    # Surface variables (in your data*.grib files)
    "sp":   "surface_pressure",
    "msl":  "mean_sea_level_pressure",
    "2t":   "2m_temperature",
    "2d":   "2m_dewpoint_temperature",
    "10u":  "10m_u_component_of_wind",
    "10v":  "10m_v_component_of_wind",
    "tp":   "total_precipitation",
    "sst":  "sea_surface_temperature",
    "siconc": "sea_ice_cover",
    "cape": "convective_available_potential_energy",
    "cin":  "convective_inhibition",
    "blh":  "boundary_layer_height",
    "tcc":  "total_cloud_cover",
    "lcc":  "low_cloud_cover",
    "mcc":  "medium_cloud_cover",
    "hcc":  "high_cloud_cover",
    "sd":   "snow_depth",
    "sde":  "snow_density",
    "stl1": "soil_temperature_level_1",
    "stl2": "soil_temperature_level_2",
    "stl3": "soil_temperature_level_3",
    "stl4": "soil_temperature_level_4",
    "swvl1":"volumetric_soil_water_layer_1",
    "swvl2":"volumetric_soil_water_layer_2",
    "swvl3":"volumetric_soil_water_layer_3",
    "swvl4":"volumetric_soil_water_layer_4",
    "e":    "evaporation",
    "ro":   "runoff",
}

# CDS long name → NeuralGCM name (fallback)
CDS_LONG_TO_NEURALGCM = {
    "u_component_of_wind":               "u_component_of_wind",
    "v_component_of_wind":               "v_component_of_wind",
    "temperature":                        "temperature",
    "specific_humidity":                  "specific_humidity",
    "geopotential":                       "geopotential",
    "specific_cloud_liquid_water_content":"specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content":   "specific_cloud_ice_water_content",
    "vertical_velocity":                  "vertical_velocity",
    "surface_pressure":                   "surface_pressure",
}


class ERA5GRIBLoader:
    """
    Loads and merges the 7 GRIB files into two clean xarray Datasets:
      - pressure_ds: all pressure-level variables merged across levels
      - surface_ds:  all single-level variables merged across months
    """

    def __init__(self, grib_dir: str, config: dict):
        self.grib_dir = Path(grib_dir)
        self.config   = config
        self.pressure_ds: Optional[xr.Dataset] = None
        self.surface_ds:  Optional[xr.Dataset] = None
        self.merged_ds:   Optional[xr.Dataset] = None

    # ── PUBLIC INTERFACE ──────────────────────────────────────

    def load_all(self, use_cache: bool = True) -> xr.Dataset:
        """
        Load and merge all 7 GRIB files.
        Returns unified Dataset with all variables.
        """
        merged_cache = Path(
            self.config["data"]["cache_merged_zarr"])

        if use_cache and merged_cache.exists():
            logger.info(f"Loading merged cache: {merged_cache}")
            self.merged_ds = xr.open_zarr(
                str(merged_cache), chunks=None)
            self._log_dataset_summary(self.merged_ds, "merged")
            return self.merged_ds

        # Load pressure levels
        self.pressure_ds = self._load_pressure_levels(use_cache)
        # Load surface levels
        self.surface_ds  = self._load_surface_levels(use_cache)
        # Merge into one dataset
        self.merged_ds   = self._merge_pressure_and_surface()
        # Cache
        merged_cache.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving merged cache: {merged_cache}")
        self.merged_ds.to_zarr(str(merged_cache), mode="w")
        logger.success("Merged cache saved.")
        return self.merged_ds

    def get_available_times(self) -> List[pd.Timestamp]:
        """Return sorted list of all timesteps in the dataset."""
        if self.merged_ds is None:
            raise RuntimeError("Call load_all() first")
        if "time" not in self.merged_ds.coords:
            return [pd.Timestamp("2024-01-01")]
        times = pd.to_datetime(
            self.merged_ds.time.values).tolist()
        return sorted(times)

    def get_init_state(
        self,
        t: pd.Timestamp,
        pressure_vars: List[str],
        surface_vars: List[str],
    ) -> xr.Dataset:
        """
        Return one time slice suitable for NeuralGCM encoding.
        Includes all required pressure-level and surface variables.
        """
        if self.merged_ds is None:
            raise RuntimeError("Call load_all() first")
        slc = self.merged_ds.sel(
            time=t, method="nearest").compute()
        return slc

    # ── PRESSURE-LEVEL LOADING ────────────────────────────────

    def _load_pressure_levels(
        self, use_cache: bool
    ) -> xr.Dataset:
        """
        Load pressurelevel1/2/3.grib and merge along level dim.
        Result: Dataset with dims (time, level, latitude, longitude)
        """
        pl_cache = Path(
            self.config["data"]["cache_pressure_zarr"])

        if use_cache and pl_cache.exists():
            logger.info(f"Loading pressure cache: {pl_cache}")
            ds = xr.open_zarr(str(pl_cache), chunks=None)
            self._log_dataset_summary(ds, "pressure-level")
            return ds

        pl_files = [
            self.grib_dir / f
            for f in self.config["data"]["pressure_level_files"]
        ]
        logger.info(
            f"Loading {len(pl_files)} pressure-level GRIB files...")

        datasets = []
        for f in pl_files:
            if not f.exists():
                raise FileNotFoundError(
                    f"Pressure-level file not found: {f}\n"
                    f"Expected at: {f.absolute()}")
            logger.info(f"  Reading: {f.name}")
            ds = self._read_grib_file(f, file_type="pressure")
            if ds is not None:
                self._log_dataset_summary(
                    ds, f.name, brief=True)
                datasets.append(ds)

        if not datasets:
            raise RuntimeError(
                "Could not load any pressure-level GRIB files")

        # Merge along level coordinate
        # The 3 files cover different level ranges — concat works
        # because time and spatial coords are identical
        logger.info(
            "Merging pressure-level files along level dimension...")
        try:
            # Try merge first (if levels don't overlap)
            merged = xr.merge(
                datasets, compat="override", join="outer")
        except Exception as e:
            logger.warning(
                f"merge failed ({e}), trying concat on level...")
            try:
                merged = xr.concat(datasets, dim="level")
                merged = merged.sortby("level")
            except Exception as e2:
                logger.warning(
                    f"concat on level failed ({e2}), "
                    "trying concat on time and taking unique levels...")
                all_levs = {}
                for ds in datasets:
                    for lev in ds.level.values:
                        if lev not in all_levs:
                            all_levs[lev] = ds.sel(level=[lev])
                parts = [all_levs[l]
                         for l in sorted(all_levs.keys())]
                merged = xr.concat(parts, dim="level")

        merged = self._standardise_coords(merged)
        merged = self._rename_vars(merged)

        # Verify all expected levels are present
        expected = set(self.config["data"]["pressure_levels"])
        found    = set(merged.level.values.tolist()) \
            if "level" in merged.coords else set()
        missing  = expected - found
        if missing:
            logger.warning(
                f"Expected pressure levels not found: "
                f"{sorted(missing)}")
        else:
            logger.success(
                f"All {len(expected)} pressure levels present")

        # Cache
        pl_cache.parent.mkdir(parents=True, exist_ok=True)
        merged.to_zarr(str(pl_cache), mode="w")
        logger.success(f"Pressure-level cache saved: {pl_cache}")
        return merged

    # ── SURFACE-LEVEL LOADING ─────────────────────────────────

    def _load_surface_levels(
        self, use_cache: bool
    ) -> xr.Dataset:
        """
        Load data1/2/3/4.grib and merge along time dimension.
        Result: Dataset with dims (time, latitude, longitude)
        """
        sl_cache = Path(
            self.config["data"]["cache_surface_zarr"])

        if use_cache and sl_cache.exists():
            logger.info(f"Loading surface cache: {sl_cache}")
            ds = xr.open_zarr(str(sl_cache), chunks=None)
            self._log_dataset_summary(ds, "surface-level")
            return ds

        sl_files = [
            self.grib_dir / f
            for f in self.config["data"]["single_level_files"]
        ]
        logger.info(
            f"Loading {len(sl_files)} single-level GRIB files...")

        datasets = []
        for f in sl_files:
            if not f.exists():
                raise FileNotFoundError(
                    f"Single-level file not found: {f}")
            logger.info(f"  Reading: {f.name}")
            ds = self._read_grib_file(f, file_type="surface")
            if ds is not None:
                self._log_dataset_summary(
                    ds, f.name, brief=True)
                datasets.append(ds)

        if not datasets:
            raise RuntimeError(
                "Could not load any single-level GRIB files")

        # Concatenate along time (4 files = 4 quarters of 2024)
        logger.info(
            "Concatenating single-level files along time...")
        try:
            merged = xr.concat(
                datasets, dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
            )
            merged = merged.sortby("time")
            # Remove duplicate times (from file boundaries)
            _, idx = np.unique(merged.time.values,
                               return_index=True)
            merged = merged.isel(time=idx)
        except Exception as e:
            logger.warning(f"concat failed ({e}), trying merge...")
            merged = xr.merge(
                datasets, compat="override", join="outer")

        merged = self._standardise_coords(merged)
        merged = self._rename_vars(merged)

        # Derive log_surface_pressure from surface_pressure
        if "surface_pressure" in merged.data_vars:
            sp = merged["surface_pressure"]
            # Ensure SP is in Pa (typical range 85000-106000 Pa)
            sp_mean = float(sp.mean())
            if sp_mean < 2000:
                logger.info(
                    f"  SP mean={sp_mean:.1f} — appears to be "
                    "in hPa, converting to Pa x100")
                sp = sp * 100.0
            logger.info(
                f"  Deriving log_surface_pressure "
                f"(SP mean={float(sp.mean()):.1f} Pa)")
            merged["log_surface_pressure"] = np.log(
                sp.clip(min=1.0))
            logger.success("  log_surface_pressure derived")
        else:
            logger.warning(
                "  surface_pressure not found in single-level "
                "data — log_surface_pressure will be missing")

        # Cache
        sl_cache.parent.mkdir(parents=True, exist_ok=True)
        merged.to_zarr(str(sl_cache), mode="w")
        logger.success(f"Surface-level cache saved: {sl_cache}")
        return merged

    # ── MERGE PRESSURE + SURFACE ──────────────────────────────

    def _merge_pressure_and_surface(self) -> xr.Dataset:
        """
        Combine pressure-level and surface datasets.
        Aligns on time coordinate.
        """
        logger.info(
            "Merging pressure-level and surface datasets...")

        # Find common times
        pl_times = set(
            pd.to_datetime(self.pressure_ds.time.values))
        sl_times = set(
            pd.to_datetime(self.surface_ds.time.values))
        common   = sorted(pl_times & sl_times)

        if not common:
            logger.warning(
                "No common times between pressure and surface "
                "datasets! Check that all files cover same dates.")
            # Fall back to using just pressure levels
            return self.pressure_ds

        logger.info(
            f"  Common time steps: {len(common)} "
            f"({common[0].strftime('%Y-%m-%d')} to "
            f"{common[-1].strftime('%Y-%m-%d')})")

        common_pd = pd.DatetimeIndex(common)
        pl = self.pressure_ds.sel(
            time=common_pd, method="nearest")
        sl = self.surface_ds.sel(
            time=common_pd, method="nearest")

        # Merge — surface has no level dim, pressure has level dim
        try:
            merged = xr.merge([pl, sl],
                               compat="override", join="inner")
        except Exception as e:
            logger.warning(
                f"Merge error: {e} — using pressure levels only")
            merged = pl

        logger.success(
            f"Merged dataset: {dict(merged.sizes)} | "
            f"vars={list(merged.data_vars)[:10]}...")
        return merged

    # ── GRIB READING ──────────────────────────────────────────

    def _read_grib_file(
        self,
        path: Path,
        file_type: str = "pressure",
    ) -> Optional[xr.Dataset]:
        """
        Read a single GRIB file using cfgrib.
        Handles both single-level and pressure-level files.
        Returns xarray Dataset or None on failure.
        """
        try:
            import cfgrib
        except ImportError:
            raise ImportError(
                "cfgrib not installed. Run: "
                "pip install cfgrib eccodes")

        # Strategy: try multiple cfgrib open modes
        # cfgrib.open_datasets handles multi-message GRIB files
        all_ds = []

        # Primary: open all messages
        try:
            dsets = cfgrib.open_datasets(
                str(path),
                backend_kwargs={
                    "indexpath": "",
                    "errors":   "ignore",
                    "squeeze":  False,
                },
            )
            all_ds.extend(dsets)
            logger.info(f"    cfgrib.open_datasets: {len(dsets)} sub-datasets")
        except Exception as e:
            logger.warning(
                f"  cfgrib.open_datasets failed: {e}")

        # Fallback: xarray with cfgrib engine
        if not all_ds:
            try:
                ds = xr.open_dataset(
                    str(path),
                    engine="cfgrib",
                    backend_kwargs={"errors": "ignore"},
                )
                all_ds.append(ds)
                logger.info("    xr.open_dataset cfgrib: success")
            except Exception as e:
                logger.warning(
                    f"  xr.open_dataset cfgrib failed: {e}")

        # Second fallback: open with specific level type filter
        if not all_ds:
            level_type = ("isobaricInhPa"
                          if file_type == "pressure"
                          else "surface")
            try:
                ds = xr.open_dataset(
                    str(path),
                    engine="cfgrib",
                    backend_kwargs={
                        "filter_by_keys": {
                            "typeOfLevel": level_type},
                        "errors": "ignore",
                    },
                )
                all_ds.append(ds)
                logger.info(f"    level-type filter ({level_type}): success")
            except Exception as e:
                logger.error(
                    f"  All read strategies failed for "
                    f"{path.name}: {e}")
                return None

        if not all_ds:
            return None

        # For pressure-level files, filter to only datasets that have
        # a pressure level coordinate (skip surface-type sub-datasets
        # that cfgrib sometimes extracts from pressure-level files)
        if file_type == "pressure":
            pressure_ds = []
            for ds in all_ds:
                has_level = any(
                    c.lower() in ("isobaricinhpa", "level", "plev",
                                  "pressure_level")
                    for c in list(ds.coords) + list(ds.dims)
                )
                if has_level:
                    pressure_ds.append(ds)
            if pressure_ds:
                all_ds = pressure_ds
                logger.info(
                    f"    Filtered to {len(all_ds)} pressure-level sub-datasets")

        # Merge all sub-datasets from this file
        if len(all_ds) == 1:
            return all_ds[0]

        try:
            return xr.merge(all_ds, compat="override",
                            join="outer")
        except Exception:
            # Return largest dataset
            return max(all_ds,
                       key=lambda d: len(d.data_vars))

    # ── STANDARDISATION ───────────────────────────────────────

    def _standardise_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardise coordinate names to ERA5 conventions."""
        rename = {}
        for c in list(ds.coords) + list(ds.dims):
            cl = c.lower()
            if cl in ("lat", "latitude_0") and \
                    "latitude" not in ds.coords:
                rename[c] = "latitude"
            elif cl in ("lon", "longitude_0") and \
                    "longitude" not in ds.coords:
                rename[c] = "longitude"
            elif cl in ("isobaricinhpa", "plev",
                         "pressure_level") and \
                    "level" not in ds.coords:
                rename[c] = "level"
        if rename:
            ds = ds.rename(rename)

        # Ensure latitude is descending (90 -> -90)
        if "latitude" in ds.coords:
            if len(ds.latitude) > 1 and \
                    ds.latitude.values[0] < ds.latitude.values[-1]:
                ds = ds.isel(latitude=slice(None, None, -1))

        # Ensure longitude is 0-360
        if "longitude" in ds.coords:
            lon = ds.longitude.values
            if lon.min() < 0:
                ds = ds.assign_coords(
                    longitude=(ds.longitude % 360))
                ds = ds.sortby("longitude")
        return ds

    def _rename_vars(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename variables to NeuralGCM naming conventions."""
        rename = {}
        for v in list(ds.data_vars):
            # Try shortName first
            mapped = GRIB_SHORT_TO_NEURALGCM.get(v)
            if not mapped:
                # Try long name
                mapped = CDS_LONG_TO_NEURALGCM.get(v)
            if not mapped:
                # Try attrs
                short = ds[v].attrs.get("GRIB_shortName", "")
                mapped = GRIB_SHORT_TO_NEURALGCM.get(short)
            if not mapped:
                long_name = ds[v].attrs.get(
                    "long_name",
                    ds[v].attrs.get("GRIB_name", ""))
                # Normalise long name to snake_case
                normed = long_name.lower().replace(" ", "_")
                mapped = CDS_LONG_TO_NEURALGCM.get(normed)
            if mapped and mapped != v:
                rename[v] = mapped

        if rename:
            # Avoid conflicts: only rename if target doesn't exist
            safe_rename = {k: v for k, v in rename.items()
                          if v not in ds.data_vars}
            if safe_rename:
                ds = ds.rename(safe_rename)
                logger.info(f"  Renamed vars: {safe_rename}")
        return ds

    # ── UTILITIES ─────────────────────────────────────────────

    def _log_dataset_summary(
        self,
        ds: xr.Dataset,
        name: str,
        brief: bool = False,
    ):
        if brief:
            logger.info(
                f"  {name}: vars={list(ds.data_vars)[:8]}... "
                f"dims={dict(ds.sizes)}")
        else:
            logger.info(f"  Dataset: {name}")
            logger.info(f"  Dims:    {dict(ds.sizes)}")
            logger.info(f"  Vars:    {list(ds.data_vars)[:15]}...")
            if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                times = pd.to_datetime(ds.time.values)
                logger.info(
                    f"  Time:    {times[0].strftime('%Y-%m-%d')}"
                    f" to {times[-1].strftime('%Y-%m-%d')}"
                    f" ({len(times)} steps)")
            if "level" in ds.coords:
                logger.info(
                    f"  Levels:  {sorted(ds.level.values.tolist())}")
