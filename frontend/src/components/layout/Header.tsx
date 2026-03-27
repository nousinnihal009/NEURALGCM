import { useState, useEffect } from "react";
import { Cloud, Wifi, WifiOff } from "lucide-react";
import { LocationSearch } from "../search/LocationSearch";
import { useForecastStore } from "../../store/forecastStore";
import { checkHealth } from "../../api/forecast";

export function Header() {
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const { forecastDays, forecastMode,
          setForecastDays, setForecastMode } = useForecastStore();

  useEffect(() => {
    const check = async () => setApiOnline(await checkHealth());
    check();
    const iv = setInterval(check, 30_000);
    return () => clearInterval(iv);
  }, []);

  return (
    <header className="h-12 flex-shrink-0 bg-bg-secondary
                       border-b border-border flex items-center
                       px-4 gap-4 z-10">
      {/* Brand */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <Cloud size={18} className="text-accent-blue" />
        <span className="font-semibold text-sm text-text-primary">
          NeuralGCM
        </span>
        <span className="text-xs text-text-muted hidden sm:inline">
          Weather
        </span>
      </div>

      {/* Search */}
      <div className="flex-1 max-w-xs">
        <LocationSearch />
      </div>

      <div className="flex items-center gap-3 ml-auto flex-shrink-0">
        {/* Days selector */}
        <div className="flex items-center gap-1.5 text-xs">
          <span className="text-text-muted hidden sm:inline">Days:</span>
          {[3, 5, 7, 10].map((d) => (
            <button
              key={d}
              onClick={() => setForecastDays(d)}
              className={`w-7 h-6 rounded text-xs font-medium transition-colors
                ${forecastDays === d
                  ? "bg-accent-blue text-white"
                  : "text-text-muted hover:text-text-primary"
                }`}
            >
              {d}
            </button>
          ))}
        </div>

        {/* Mode selector */}
        <div className="flex items-center gap-1 text-xs bg-bg-tertiary
                        rounded-md p-0.5 border border-border">
          {(["historical", "realtime"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setForecastMode(m)}
              className={`px-2 py-1 rounded transition-colors capitalize
                ${forecastMode === m
                  ? "bg-accent-blue/20 text-accent-blue"
                  : "text-text-muted hover:text-text-secondary"
                }`}
            >
              {m}
            </button>
          ))}
        </div>

        {/* API status */}
        <div
          className={`flex items-center gap-1.5 text-xs
            ${apiOnline === true  ? "text-accent-green"
            : apiOnline === false ? "text-accent-orange"
            : "text-text-muted"}`}
          title={`API ${apiOnline === true ? "online" : "offline"}`}
        >
          {apiOnline
            ? <Wifi size={13} />
            : <WifiOff size={13} />}
          <span className="hidden sm:inline">
            {apiOnline === null ? "…" : apiOnline ? "Online" : "Offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
