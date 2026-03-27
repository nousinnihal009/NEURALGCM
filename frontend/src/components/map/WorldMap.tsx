import { useCallback, useRef, useState } from "react";
import Map, {
  Marker, NavigationControl, ScaleControl,
  type MapRef, type MapLayerMouseEvent
} from "react-map-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { useForecastStore } from "../../store/forecastStore";
import { useForecast } from "../../hooks/useForecast";
import { ForecastMarker } from "./ForecastMarker";

const MAPBOX_TOKEN  = import.meta.env.VITE_MAPBOX_TOKEN || "";
const MAP_STYLE_DARK = MAPBOX_TOKEN
  ? "mapbox://styles/mapbox/dark-v11"
  : "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export function WorldMap() {
  const mapRef = useRef<MapRef>(null);
  const {
    selectedLocation, isPolling,
    mapCenter, mapZoom, setMapView
  } = useForecastStore();
  const { triggerForecast } = useForecast();
  const [cursor, setCursor] = useState("crosshair");

  const handleClick = useCallback(
    async (e: MapLayerMouseEvent) => {
      if (isPolling) return;
      const { lat, lng } = e.lngLat;
      await triggerForecast({
        lat,
        lon: lng,
        name: `${lat.toFixed(4)}°N, ${lng.toFixed(4)}°E`,
      });
    },
    [isPolling, triggerForecast]
  );

  const handleMove = useCallback(
    (e: { viewState: { longitude: number; latitude: number; zoom: number } }) => {
      setMapView(
        [e.viewState.longitude, e.viewState.latitude],
        e.viewState.zoom
      );
    },
    [setMapView]
  );

  return (
    <div className="relative w-full h-full">
      <Map
        ref={mapRef}
        mapboxAccessToken={MAPBOX_TOKEN || undefined}
        mapStyle={MAP_STYLE_DARK}
        initialViewState={{
          longitude: mapCenter[0],
          latitude:  mapCenter[1],
          zoom:      mapZoom,
        }}
        cursor={isPolling ? "wait" : cursor}
        onClick={handleClick}
        onMove={handleMove}
        onMouseEnter={() => setCursor(isPolling ? "wait" : "crosshair")}
        style={{ width: "100%", height: "100%" }}
        attributionControl={false}
      >
        <NavigationControl position="top-right" />
        <ScaleControl position="bottom-left" />

        {selectedLocation && (
          <Marker
            latitude={selectedLocation.lat}
            longitude={selectedLocation.lon}
            anchor="bottom"
          >
            <ForecastMarker isPolling={isPolling} />
          </Marker>
        )}
      </Map>

      {!selectedLocation && (
        <div
          className="absolute bottom-8 left-1/2 -translate-x-1/2
                     bg-bg-secondary/90 border border-border rounded-lg
                     px-4 py-2 text-sm text-text-secondary
                     pointer-events-none select-none"
        >
          Click anywhere on the map to get a 5-day NeuralGCM forecast
        </div>
      )}

      {isPolling && (
        <div
          className="absolute top-4 left-1/2 -translate-x-1/2
                     bg-bg-secondary border border-accent-blue/30
                     rounded-lg px-4 py-2 text-sm text-accent-blue
                     flex items-center gap-2 shadow-lg"
        >
          <div className="w-2 h-2 bg-accent-blue rounded-full
                          animate-pulse-slow" />
          Running NeuralGCM inference…
        </div>
      )}
    </div>
  );
}
