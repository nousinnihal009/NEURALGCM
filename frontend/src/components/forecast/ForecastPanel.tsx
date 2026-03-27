import { motion, AnimatePresence } from "framer-motion";
import { MapPin, Clock, Cpu, AlertCircle, CheckCircle2 } from "lucide-react";
import { useForecastStore } from "../../store/forecastStore";
import { SummaryCards }      from "./SummaryCards";
import { TemperatureChart }  from "./TemperatureChart";
import { HumidityChart }     from "./HumidityChart";
import { WindChart }         from "./WindChart";
import { TPWChart }          from "./TPWChart";
import { StabilityChart }    from "./StabilityChart";
import { Z500Chart }         from "./Z500Chart";
import { PressureChart }     from "./PressureChart";
import { ForecastTable }     from "./ForecastTable";
import { ProgressBar }       from "../ui/ProgressBar";

export function ForecastPanel() {
  const {
    currentForecast, selectedLocation,
    isPolling, pollProgress,
  } = useForecastStore();

  // Empty state
  if (!selectedLocation && !currentForecast) {
    return (
      <div className="h-full flex flex-col items-center justify-center
                      text-center px-8 text-text-muted">
        <div className="text-4xl mb-4">🌍</div>
        <p className="text-sm font-medium text-text-secondary mb-2">
          Click anywhere on the map
        </p>
        <p className="text-xs leading-relaxed">
          NeuralGCM will generate a 5-day atmospheric forecast for any
          location on Earth, initialised from ERA5 reanalysis data.
        </p>
        <div className="mt-6 text-xs text-text-muted space-y-1">
          <p>Temperature · Humidity · Wind · Pressure</p>
          <p>Geopotential · Precipitable Water · Stability</p>
        </div>
      </div>
    );
  }

  // Loading / polling state
  if (isPolling && !currentForecast) {
    return (
      <div className="h-full flex flex-col">
        {selectedLocation && (
          <div className="p-4 border-b border-border">
            <div className="flex items-center gap-2 text-sm font-medium
                            text-text-primary">
              <MapPin size={14} className="text-accent-orange" />
              {selectedLocation.name}
            </div>
            <p className="text-xs text-text-muted mt-1">
              {selectedLocation.lat.toFixed(4)}°N,{" "}
              {selectedLocation.lon.toFixed(4)}°E
            </p>
          </div>
        )}
        <div className="flex-1 flex flex-col items-center justify-center
                        px-8 text-center">
          <div className="w-full max-w-xs mb-6">
            <ProgressBar value={pollProgress} />
          </div>
          <p className="text-sm font-medium text-text-secondary mb-1">
            Running NeuralGCM inference…
          </p>
          <p className="text-xs text-text-muted">
            deterministic_2_8_deg · 5-day forecast
          </p>
          <div className="mt-4 text-xs text-text-muted space-y-0.5">
            <p>Loading ERA5 · Regridding 0.25°→2.8°</p>
            <p>Encoding state · Unrolling forecast</p>
          </div>
        </div>
      </div>
    );
  }

  if (!currentForecast) return null;
  const fc = currentForecast;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={fc.job_id}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.35 }}
        className="h-full flex flex-col overflow-hidden"
      >
        {/* Header */}
        <div className="p-4 border-b border-border flex-shrink-0">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <MapPin size={14} className="text-accent-orange flex-shrink-0" />
                <h2 className="font-semibold text-sm text-text-primary truncate">
                  {fc.location_name}
                </h2>
                {fc.is_cached && (
                  <span className="text-xs bg-accent-blue/10 text-accent-blue
                                   border border-accent-blue/20 px-1.5 py-0.5
                                   rounded-full flex-shrink-0">
                    cached
                  </span>
                )}
              </div>
              <p className="text-xs text-text-muted mt-0.5">
                {fc.model_lat?.toFixed(2)}°N, {fc.model_lon?.toFixed(2)}°E
                {" · "}model grid point
              </p>
            </div>

            {/* Sanity badge */}
            <div className="flex-shrink-0 ml-2">
              {fc.sanity_ok === true && (
                <CheckCircle2 size={16} className="text-accent-green" />
              )}
              {fc.sanity_ok === false && (
                <AlertCircle size={16} className="text-accent-orange" />
              )}
            </div>
          </div>

          {/* Meta row */}
          <div className="flex items-center gap-3 mt-2 text-xs text-text-muted">
            <span className="flex items-center gap-1">
              <Clock size={11} />
              Init: {fc.init_time_utc
                ? new Date(fc.init_time_utc).toLocaleDateString("en-GB", {
                    day: "numeric", month: "short", year: "numeric",
                    timeZone: "UTC"
                  })
                : "—"}
            </span>
            <span className="flex items-center gap-1">
              <Cpu size={11} />
              {fc.mode_used ?? "—"}
            </span>
            {fc.elapsed_seconds && (
              <span>{fc.elapsed_seconds.toFixed(1)}s</span>
            )}
          </div>

          {/* Model badge */}
          <div className="mt-2">
            <span className="text-xs font-mono bg-bg-tertiary
                             border border-border rounded px-2 py-0.5
                             text-text-muted">
              NeuralGCM · deterministic_2_8_deg
            </span>
          </div>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          <SummaryCards forecast={fc} />

          <div className="space-y-6">
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <TemperatureChart daily={fc.daily} />
            </div>
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <WindChart daily={fc.daily} />
            </div>
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <HumidityChart daily={fc.daily} />
            </div>
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <TPWChart daily={fc.daily} />
            </div>
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <Z500Chart daily={fc.daily} />
            </div>
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <PressureChart daily={fc.daily} />
            </div>
            <div className="bg-bg-secondary rounded-lg p-4 border border-border">
              <StabilityChart daily={fc.daily} />
            </div>
          </div>

          <div className="bg-bg-secondary rounded-lg p-4 border border-border">
            <h3 className="text-xs font-semibold text-text-secondary
                           uppercase tracking-wide mb-2">
              Forecast Summary Table
            </h3>
            <ForecastTable forecast={fc} />
          </div>

          <p className="text-xs text-text-muted text-center pb-4">
            {fc.paper_reference} ·{" "}
            <a
              href="https://arxiv.org/abs/2311.07222"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-blue hover:underline"
            >
              arXiv:2311.07222
            </a>
          </p>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
