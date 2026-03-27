"""
API Integration Tests
=====================
Tests all forecast endpoints against an in-memory test database.
Uses httpx AsyncClient.
"""

import pytest
import pytest_asyncio
import uuid

@pytest.mark.anyio
async def test_health_endpoint(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data

@pytest.mark.anyio
async def test_ready_endpoint(client):
    r = await client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert "checks" in data

@pytest.mark.anyio
async def test_submit_forecast_validation(client):
    r = await client.post("/api/v1/forecast", json={})
    assert r.status_code == 422

    # Invalid lat/lon
    r = await client.post("/api/v1/forecast", json={
        "location_name": "Test", "lat": 999, "lon": 0})
    assert r.status_code == 422

@pytest.mark.anyio
async def test_submit_and_list_forecast(client):
    r = await client.post("/api/v1/forecast", json={
        "location_name": "Chennai, India",
        "lat": 13.0827,
        "lon": 80.2707,
        "days": 5,
        "mode": "historical",
        "init_date": "2020-06-01",
    })
    assert r.status_code in (202, 200)
    data = r.json()
    assert "job_id" in data
    job_id = data["job_id"]

    # Poll status
    r2 = await client.get(f"/api/v1/forecast/{job_id}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "pending"

    # List overall
    r3 = await client.get("/api/v1/forecasts")
    assert r3.status_code == 200
    assert r3.json()["total"] >= 1
    items = r3.json()["items"]
    assert any(i["job_id"] == job_id for i in items)

@pytest.mark.anyio
async def test_delete_forecast(client):
    r = await client.post("/api/v1/forecast", json={
        "location_name": "Delete Test",
        "lat": 0.0,
        "lon": 0.0,
        "days": 2,
        "mode": "realtime"
    })
    job_id = r.json()["job_id"]

    r_del = await client.delete(f"/api/v1/forecast/{job_id}")
    assert r_del.status_code == 204

    r_get = await client.get(f"/api/v1/forecast/{job_id}")
    assert r_get.status_code == 404

@pytest.mark.anyio
async def test_location_geom_stored_as_wkt(client, db_session):
    """
    Verify that _point_geom() stores a string representation that
    round-trips through the column without data loss.
    In SQLite tests the geom column is String(255); we check that
    the stored value contains the coordinate pair, not a Python repr.
    """
    from api.models.location import Location
    from sqlalchemy import select, text

    create = await client.post("/api/v1/locations", json={
        "name": "Geom Test Location",
        "lat": 13.0827,
        "lon": 80.2707,
    })
    assert create.status_code == 201
    loc_id = create.json()["id"]

    # Fetch directly from DB to inspect the raw geom value
    # Avoid calling UUID() on loc_id if UUID import not at top:
    result = await db_session.execute(
        select(Location).where(Location.id == loc_id)
    )
    loc = result.scalar_one_or_none()
    assert loc is not None, "Location not found in DB after creation"

    geom_val = loc.geom
    assert geom_val is not None, (
        "geom column is NULL — _point_geom() did not write a value")

    geom_str = str(geom_val)
    assert "80.2707" in geom_str, (
        f"Longitude 80.2707 not found in stored geom value: {geom_str!r}\n"
        "This indicates WKTElement.__repr__ was stored instead of WKT.")
    assert "13.0827" in geom_str, (
        f"Latitude 13.0827 not found in stored geom value: {geom_str!r}")
    assert "<" not in geom_str, (
        f"Python object repr leaked into geom column: {geom_str!r}\n"
        "SQLAlchemy called str(WKTElement) instead of using the WKT.")


@pytest.mark.anyio
async def test_point_geom_rejects_nan():
    """_point_geom() must raise ValueError for NaN coordinates."""
    from api.routers.forecast import _point_geom
    import math
    import pytest as _pytest

    with _pytest.raises(ValueError, match="NaN"):
        _point_geom(math.nan, 13.0)

    with _pytest.raises(ValueError, match="NaN"):
        _point_geom(80.0, math.nan)
