"""
Pipeline Smoke Tests
====================
End-to-end tests that verify the pipeline modules can be imported
and basic configuration is correct. Does NOT require network/model.
"""

import pytest
from neuralgcm_weather.config import MODEL, DATA, OUTPUT, SANITY, CFG_RAW


class TestConfig:

    def test_model_config_loaded(self):
        """ModelConfig should have valid defaults."""
        assert MODEL.checkpoint == "v1/deterministic_2_8_deg.pkl"
        assert MODEL.forecast_days == 5
        assert MODEL.timestep_hours == 24

    def test_data_config_loaded(self):
        """DataConfig should have valid defaults."""
        assert DATA.mode in ("realtime", "historical")
        assert "era5" in DATA.era5_zarr.lower()

    def test_output_config_loaded(self):
        """OutputConfig should have valid defaults."""
        assert OUTPUT.save_dir == "./forecasts"
        assert OUTPUT.save_png is True
        assert OUTPUT.save_json is True

    def test_sanity_bounds_exist(self):
        """Sanity bounds should be defined for key variables."""
        assert "temperature_c" in SANITY.bounds
        assert "z500_m" in SANITY.bounds
        assert "mslp_hpa" in SANITY.bounds

    def test_scheduler_locations(self):
        """Scheduler should have at least one location configured."""
        locations = CFG_RAW["scheduler"]["locations"]
        assert len(locations) >= 1
        assert "name" in locations[0]
        assert "lat" in locations[0]
        assert "lon" in locations[0]


class TestImports:

    def test_import_config(self):
        from neuralgcm_weather import config
        assert hasattr(config, 'MODEL')

    def test_import_ecmwf_loader(self):
        from neuralgcm_weather.data import ecmwf_loader
        assert hasattr(ecmwf_loader, 'load_realtime_init_state')

    def test_import_era5_loader(self):
        from neuralgcm_weather.data import era5_loader
        assert hasattr(era5_loader, 'open_era5')

    def test_import_extractor(self):
        from neuralgcm_weather.model import extractor
        assert hasattr(extractor, 'VariableExtractor')
        assert hasattr(extractor, 'ForecastPoint')

    def test_import_validator(self):
        from neuralgcm_weather.output import validator
        assert hasattr(validator, 'validate_forecast')

    def test_import_writer(self):
        from neuralgcm_weather.output import writer
        assert hasattr(writer, 'save_forecast')
