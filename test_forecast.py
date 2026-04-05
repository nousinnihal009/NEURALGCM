"""Quick test to hit the forecast endpoint and see the actual error."""
import requests
import json

url = "http://localhost:8000/api/v1/forecast"
payload = {
    "location_name": "Chennai, India",
    "lat": 13.0827,
    "lon": 80.2707,
    "days": 5,
    "mode": "historical",
    "init_date": "2020-06-01"
}

try:
    resp = requests.post(url, json=payload, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Headers: {dict(resp.headers)}")
    try:
        print(f"Body: {json.dumps(resp.json(), indent=2)}")
    except:
        print(f"Body (raw): {resp.text[:2000]}")
except Exception as e:
    print(f"Request error: {e}")
