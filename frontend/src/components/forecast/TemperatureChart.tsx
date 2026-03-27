import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine
} from "recharts";
import type { DailyForecast } from "../../api/types";
import { CHART_COLORS } from "../../utils/colours";
import { fmtDate } from "../../utils/units";

interface Props { daily: DailyForecast[]; }

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-secondary border border-border rounded-lg
                    p-3 text-xs shadow-lg">
      <p className="text-text-secondary mb-2 font-medium">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }} className="mb-1">
          {p.name}: {p.value != null ? `${p.value.toFixed(1)}°C` : "—"}
        </p>
      ))}
    </div>
  );
};

export function TemperatureChart({ daily }: Props) {
  const data = daily.map((d) => ({
    date:  fmtDate(d.date),
    t850:  d.temperature_c_850,
    t500:  d.temperature_c_500,
  }));

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-secondary uppercase
                     tracking-wide mb-3">
        Temperature (°C) · 850 hPa near-surface &amp; 500 hPa mid-atmosphere
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data}
          margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3"
            stroke="#30363d" strokeOpacity={0.6} />
          <XAxis dataKey="date" tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <YAxis unit="°C" tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} width={48} />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: "11px", color: "#8B949E" }} />
          <ReferenceLine y={35} stroke="#F78166" strokeDasharray="4 4"
            strokeOpacity={0.5} label={{ value: "35°C heat", fill: "#F78166", fontSize: 10 }} />
          <ReferenceLine y={0} stroke="#58A6FF" strokeDasharray="4 4"
            strokeOpacity={0.4} label={{ value: "0°C freeze", fill: "#58A6FF", fontSize: 10 }} />
          <Line
            type="monotone" dataKey="t850"
            name="850 hPa (near-surface)"
            stroke={CHART_COLORS.temperature850}
            strokeWidth={2.5} dot={{ r: 4, fill: CHART_COLORS.temperature850 }}
            activeDot={{ r: 6 }}
          />
          <Line
            type="monotone" dataKey="t500"
            name="500 hPa (mid-atm)"
            stroke={CHART_COLORS.temperature500}
            strokeWidth={2} strokeDasharray="6 3"
            dot={{ r: 3, fill: CHART_COLORS.temperature500 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
