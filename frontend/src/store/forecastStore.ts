import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ForecastResult, MapLocation } from "../api/types";

interface ForecastStore {
  selectedLocation:    MapLocation | null;
  setSelectedLocation: (loc: MapLocation | null) => void;

  currentJobId:    string | null;
  setCurrentJobId: (id: string | null) => void;

  currentForecast:    ForecastResult | null;
  setCurrentForecast: (fc: ForecastResult | null) => void;

  isPolling:    boolean;
  setIsPolling: (v: boolean) => void;
  pollProgress: number;
  setPollProgress: (v: number) => void;

  history:    ForecastResult[];
  addToHistory:    (fc: ForecastResult) => void;
  clearHistory: () => void;

  mapCenter: [number, number];
  mapZoom:   number;
  setMapView: (center: [number, number], zoom: number) => void;

  forecastDays: number;
  forecastMode: "realtime" | "historical";
  setForecastDays: (d: number) => void;
  setForecastMode: (m: "realtime" | "historical") => void;
}

export const useForecastStore = create<ForecastStore>()(
  persist(
    (set, get) => ({
      selectedLocation:    null,
      setSelectedLocation: (loc) => set({ selectedLocation: loc }),

      currentJobId:    null,
      setCurrentJobId: (id) => set({ currentJobId: id }),

      currentForecast:    null,
      setCurrentForecast: (fc) => set({ currentForecast: fc }),

      isPolling:    false,
      setIsPolling: (v) => set({ isPolling: v }),
      pollProgress: 0,
      setPollProgress: (v) => set({ pollProgress: v }),

      history: [],
      addToHistory: (fc) => {
        const prev = get().history.filter(h => h.job_id !== fc.job_id);
        set({ history: [fc, ...prev].slice(0, 20) });
      },
      clearHistory: () => set({ history: [] }),

      mapCenter: [78.9629, 20.5937],
      mapZoom:   4,
      setMapView: (center, zoom) => set({ mapCenter: center, mapZoom: zoom }),

      forecastDays: 5,
      forecastMode: "historical",
      setForecastDays: (d) => set({ forecastDays: d }),
      setForecastMode: (m) => set({ forecastMode: m }),
    }),
    {
      name: "neuralgcm-store",
      partialize: (state) => ({
        history:      state.history,
        forecastDays: state.forecastDays,
        forecastMode: state.forecastMode,
        mapCenter:    state.mapCenter,
        mapZoom:      state.mapZoom,
      }),
    }
  )
);
