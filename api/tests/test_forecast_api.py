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
