"""Quick test to hit the forecast endpoint and see the actual error traceback."""
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
    try:
        data = resp.json()
        if "traceback" in data:
            print(f"\n=== ERROR ===\n{data['detail']}\n")
            print(f"=== TRACEBACK ===\n{data['traceback']}")
        else:
            print(json.dumps(data, indent=2))
    except:
        print(f"Body: {resp.text[:3000]}")
except Exception as e:
    print(f"Request error: {e}")
