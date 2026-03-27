import { useCallback, useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "react-hot-toast";
import {
  submitForecast, pollForecast, reverseGeocode
} from "../api/forecast";
import { useForecastStore } from "../store/forecastStore";
import type { MapLocation } from "../api/types";

const POLL_INTERVAL_MS = 3000;
const MAX_POLL_MS      = 120_000;

export function useForecast() {
  const {
    setCurrentJobId, setCurrentForecast, setIsPolling,
    setPollProgress, addToHistory, forecastDays, forecastMode,
    setSelectedLocation,
  } = useForecastStore();

  const pollTimer   = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollStart   = useRef<number>(0);
  const queryClient = useQueryClient();

  useEffect(() => () => {
    if (pollTimer.current) clearInterval(pollTimer.current);
  }, []);

  const startPolling = useCallback((jobId: string) => {
    setIsPolling(true);
    setPollProgress(5);
    pollStart.current = Date.now();

    pollTimer.current = setInterval(async () => {
      const elapsed = Date.now() - pollStart.current;
      const progress = Math.min(95, (elapsed / MAX_POLL_MS) * 100);
      setPollProgress(progress);

      if (elapsed > MAX_POLL_MS) {
        clearInterval(pollTimer.current!);
        setIsPolling(false);
        toast.error("Forecast timed out. Please try again.");
        return;
      }

      try {
        const result = await pollForecast(jobId);

        if (result.status === "complete" || result.status === "cached") {
          clearInterval(pollTimer.current!);
          setCurrentForecast(result);
          addToHistory(result);
          setIsPolling(false);
          setPollProgress(100);
          queryClient.invalidateQueries({ queryKey: ["forecasts"] });
          const cached = result.is_cached ? " (cached)" : "";
          toast.success(
            `Forecast ready for ${result.location_name}${cached}`);
        }

        if (result.status === "failed") {
          clearInterval(pollTimer.current!);
          setIsPolling(false);
          toast.error(`Forecast failed: ${result.error || "Unknown error"}`);
        }
      } catch (e) {
        console.warn("Poll error:", e);
      }
    }, POLL_INTERVAL_MS);
  }, [setIsPolling, setPollProgress, setCurrentForecast, addToHistory, queryClient]);

  const triggerForecast = useCallback(async (location: MapLocation) => {
    if (pollTimer.current) clearInterval(pollTimer.current);
    setCurrentForecast(null);
    setCurrentJobId(null);
    setSelectedLocation(location);

    let name = location.name;
    if (!name || name.startsWith("Click")) {
      name = await reverseGeocode(location.lat, location.lon);
      setSelectedLocation({ ...location, name });
    }

    const toastId = toast.loading(`Queuing forecast for ${name}…`);

    try {
      const job = await submitForecast({
        location_name: name,
        lat:  location.lat,
        lon:  location.lon,
        days: forecastDays,
        mode: forecastMode,
        init_date: forecastMode === "historical"
          ? "2020-06-01" : undefined,
      });

      toast.dismiss(toastId);
      setCurrentJobId(job.job_id);

      if (job.status === "cached") {
        const result = await pollForecast(job.job_id);
        setCurrentForecast(result);
        addToHistory(result);
        setPollProgress(100);
        toast.success(`Forecast ready for ${name} (cached)`);
      } else {
        toast.success(`Forecast queued — estimated ${job.estimated_seconds}s`);
        startPolling(job.job_id);
      }
    } catch (e: any) {
      toast.dismiss(toastId);
      toast.error(
        `Failed to queue forecast: ${e.response?.data?.detail || e.message}`
      );
    }
  }, [forecastDays, forecastMode, startPolling, setCurrentForecast, setCurrentJobId, setSelectedLocation, addToHistory, setPollProgress]);

  return { triggerForecast };
}
