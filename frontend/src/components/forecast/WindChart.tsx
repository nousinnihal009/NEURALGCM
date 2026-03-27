import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine
} from "recharts";
import type { DailyForecast } from "../../api/types";
import { CHART_COLORS } from "../../utils/colours";
import { fmtDate } from "../../utils/units";

interface Props { daily: DailyForecast[]; }

export function WindChart({ daily }: Props) {
  const data = daily.map((d) => ({
    date: fmtDate(d.date),
    w850: d.wind_speed_850,
    w500: d.wind_speed_500,
    w250: d.wind_speed_250,
  }));

  const TooltipContent = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="bg-bg-secondary border border-border rounded-lg
                      p-3 text-xs shadow-lg">
        <p className="text-text-secondary mb-2 font-medium">{label}</p>
        {payload.map((p: any) => (
          <p key={p.name} style={{ color: p.color }} className="mb-1">
            {p.name}: {p.value != null ? `${p.value.toFixed(1)} m/s` : "—"}
          </p>
        ))}
      </div>
    );
  };

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-secondary uppercase
                     tracking-wide mb-3">
        Wind Speed (m/s) · 850 hPa surface · 500 hPa steering · 250 hPa jet
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}
          margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3"
            stroke="#30363d" strokeOpacity={0.6} />
          <XAxis dataKey="date"
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <YAxis unit=" m/s" tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} width={52} />
          <Tooltip content={<TooltipContent />} />
          <Legend wrapperStyle={{ fontSize: "11px", color: "#8B949E" }} />
          <ReferenceLine y={17} stroke={CHART_COLORS.windSurface}
            strokeDasharray="4 4" strokeOpacity={0.4}
            label={{ value: "17 m/s gale", fill: "#3FB950", fontSize: 10 }} />
          <ReferenceLine y={33} stroke={CHART_COLORS.windJet}
            strokeDasharray="4 4" strokeOpacity={0.4}
            label={{ value: "33 m/s hurricane", fill: "#F78166", fontSize: 10 }} />
          <Line type="monotone" dataKey="w850"
            name="850 hPa (surface)"
            stroke={CHART_COLORS.windSurface} strokeWidth={2.5}
            dot={{ r: 4, fill: CHART_COLORS.windSurface }}
            activeDot={{ r: 6 }} />
          <Line type="monotone" dataKey="w500"
            name="500 hPa (steering)"
            stroke={CHART_COLORS.windMid} strokeWidth={2}
            strokeDasharray="6 3"
            dot={{ r: 3, fill: CHART_COLORS.windMid }} />
          <Line type="monotone" dataKey="w250"
            name="250 hPa (jet stream)"
            stroke={CHART_COLORS.windJet} strokeWidth={1.5}
            strokeDasharray="3 3"
            dot={{ r: 3, fill: CHART_COLORS.windJet }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
