import axios from "axios";
import { apiClient } from "./client";
import type {
  ForecastRequest, ForecastJobResponse, ForecastResult
} from "./types";

export async function submitForecast(
  req: ForecastRequest
): Promise<ForecastJobResponse> {
  const { data } = await apiClient.post<ForecastJobResponse>(
    "/api/v1/forecast", req);
  return data;
}

export async function pollForecast(
  jobId: string
): Promise<ForecastResult> {
  const { data } = await apiClient.get<ForecastResult>(
    `/api/v1/forecast/${jobId}`);
  return data;
}

export async function listForecasts(
  page = 1, pageSize = 20
): Promise<{ total: number; items: ForecastResult[] }> {
  const { data } = await apiClient.get("/api/v1/forecasts", {
    params: { page, page_size: pageSize },
  });
  return data;
}

export async function checkHealth(): Promise<boolean> {
  try {
    await apiClient.get("/health");
    return true;
  } catch {
    return false;
  }
}

// Reverse geocode lat/lon to city name using Nominatim (free, no key)
export async function reverseGeocode(
  lat: number, lon: number
): Promise<string> {
  try {
    const { data } = await axios.get(
      "https://nominatim.openstreetmap.org/reverse",
      {
        params: { lat, lon, format: "json" },
        headers: { "Accept-Language": "en" },
      }
    );
    const a = data.address;
    return (
      a.city || a.town || a.village || a.county ||
      a.state || a.country || `${lat.toFixed(4)}, ${lon.toFixed(4)}`
    );
  } catch {
    return `${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`;
  }
}

// Forward geocode city name to coordinates
export async function forwardGeocode(
  query: string
): Promise<Array<{ name: string; lat: number; lon: number }>> {
  try {
    const { data } = await axios.get(
      "https://nominatim.openstreetmap.org/search",
      {
        params: { q: query, format: "json", limit: 5 },
        headers: { "Accept-Language": "en" },
      }
    );
    return data.map((r: any) => ({
      name: r.display_name,
      lat:  parseFloat(r.lat),
      lon:  parseFloat(r.lon),
    }));
  } catch {
    return [];
  }
}
