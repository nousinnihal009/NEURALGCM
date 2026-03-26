"""
API Integration Tests
=====================
Tests all forecast endpoints against a test database.
Uses httpx AsyncClient for async endpoint testing.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="module")
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c


@pytest.mark.anyio
async def test_health_endpoint(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.anyio
async def test_submit_forecast_validation(client):
    # Missing required fields
    r = await client.post("/api/v1/forecast", json={})
    assert r.status_code == 422

    # Invalid lat/lon
    r = await client.post("/api/v1/forecast", json={
        "location_name": "Test", "lat": 999, "lon": 0})
    assert r.status_code == 422

    # days out of range
    r = await client.post("/api/v1/forecast", json={
        "location_name": "Test", "lat": 13.08, "lon": 80.27,
        "days": 99})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_submit_forecast_valid(client):
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
    assert "poll_url" in data
    assert data["status"] in ("pending", "cached")


@pytest.mark.anyio
async def test_get_forecast_not_found(client):
    r = await client.get(
        "/api/v1/forecast/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_list_forecasts(client):
    r = await client.get("/api/v1/forecasts")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.anyio
async def test_ready_endpoint(client):
    r = await client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert "checks" in data
    assert "redis" in data["checks"]
    assert "postgres" in data["checks"]
