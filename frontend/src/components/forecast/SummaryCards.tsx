import { Thermometer, Droplets, Wind, CloudRain } from "lucide-react";
import type { ForecastResult } from "../../api/types";
import {
  fmtTemp, fmtRH, fmtWind, fmtTPW
} from "../../utils/units";

interface Props { forecast: ForecastResult; }

export function SummaryCards({ forecast }: Props) {
  const d = forecast.daily[Math.min(1, forecast.daily.length - 1)];
  if (!d) return null;

  const cards = [
    {
      icon:  <Thermometer size={18} />,
      label: "Temperature",
      value: fmtTemp(d.temperature_c_850),
      sub:   "850 hPa · near-surface",
      color: "#F78166",
    },
    {
      icon:  <Droplets size={18} />,
      label: "Rel. Humidity",
      value: fmtRH(d.rh_850),
      sub:   "850 hPa · low-level moisture",
      color: "#58A6FF",
    },
    {
      icon:  <Wind size={18} />,
      label: "Wind Speed",
      value: fmtWind(d.wind_speed_850),
      sub:   `${d.wind_dir_compass ?? "—"} · 850 hPa`,
      color: "#3FB950",
    },
    {
      icon:  <CloudRain size={18} />,
      label: "Precip. Water",
      value: fmtTPW(d.tpw_mm),
      sub:   "Total column · rain potential",
      color: "#A5D6FF",
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-2 mb-4">
      {cards.map((c) => (
        <div
          key={c.label}
          className="bg-bg-tertiary rounded-lg p-3 border border-border
                     hover:border-border-hover transition-colors"
        >
          <div
            className="flex items-center gap-2 mb-1"
            style={{ color: c.color }}
          >
            {c.icon}
            <span className="text-xs text-text-secondary">{c.label}</span>
          </div>
          <div
            className="text-xl font-semibold"
            style={{ color: c.color }}
          >
            {c.value}
          </div>
          <div className="text-xs text-text-muted mt-0.5">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}
