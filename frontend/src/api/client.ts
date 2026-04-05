import axios from "axios";

export const apiClient = axios.create({
  // In dev, Vite proxy forwards /api/* and /health to localhost:8000
  // No baseURL needed — avoids CORS preflight issues
  baseURL: "",
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

// Add API key if set in env
apiClient.interceptors.request.use((config) => {
  const key = import.meta.env.VITE_API_KEY;
  if (key) config.headers["X-API-Key"] = key;
  return config;
});
