import { useState, useCallback, useRef } from "react";
import { Search, X } from "lucide-react";
import { forwardGeocode } from "../../api/forecast";
import { useForecast } from "../../hooks/useForecast";
import { useForecastStore } from "../../store/forecastStore";

export function LocationSearch() {
  const [query,   setQuery]   = useState("");
  const [results, setResults] = useState<
    Array<{ name: string; lat: number; lon: number }>
  >([]);
  const [open, setOpen]       = useState(false);
  const [loading, setLoading] = useState(false);
  const debounceRef           = useRef<ReturnType<typeof setTimeout>>();

  const { triggerForecast }  = useForecast();
  const { setMapView }       = useForecastStore();

  const handleInput = useCallback((v: string) => {
    setQuery(v);
    clearTimeout(debounceRef.current);
    if (v.length < 2) { setResults([]); setOpen(false); return; }
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      const r = await forwardGeocode(v);
      setResults(r);
      setOpen(r.length > 0);
      setLoading(false);
    }, 400);
  }, []);

  const handleSelect = useCallback(
    async (r: { name: string; lat: number; lon: number }) => {
      setQuery(r.name.split(",")[0]);
      setOpen(false);
      setMapView([r.lon, r.lat], 8);
      await triggerForecast({ lat: r.lat, lon: r.lon, name: r.name });
    },
    [triggerForecast, setMapView]
  );

  return (
    <div className="relative">
      <div className="flex items-center gap-2 bg-bg-secondary
                      border border-border rounded-lg px-3 h-9
                      focus-within:border-accent-blue transition-colors">
        <Search size={14} className="text-text-muted flex-shrink-0" />
        <input
          value={query}
          onChange={(e) => handleInput(e.target.value)}
          placeholder="Search location…"
          className="bg-transparent text-sm text-text-primary
                     placeholder:text-text-muted outline-none
                     flex-1 min-w-0"
        />
        {query && (
          <button
            onClick={() => { setQuery(""); setOpen(false); }}
            className="text-text-muted hover:text-text-primary"
          >
            <X size={13} />
          </button>
        )}
      </div>

      {open && (
        <div
          className="absolute top-10 left-0 right-0 z-50
                     bg-bg-secondary border border-border rounded-lg
                     shadow-lg overflow-hidden max-h-64 overflow-y-auto"
        >
          {loading && (
            <div className="px-3 py-2 text-xs text-text-muted">
              Searching…
            </div>
          )}
          {results.map((r, i) => (
            <button
              key={i}
              onClick={() => handleSelect(r)}
              className="w-full text-left px-3 py-2 text-xs
                         text-text-secondary hover:bg-bg-tertiary
                         hover:text-text-primary transition-colors
                         border-b border-border/50 last:border-0"
            >
              <span className="font-medium">
                {r.name.split(",")[0]}
              </span>
              <span className="text-text-muted ml-1">
                {r.name.split(",").slice(1, 3).join(",")}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
