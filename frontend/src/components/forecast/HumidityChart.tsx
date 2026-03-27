import {
  ResponsiveContainer, ComposedChart, Bar, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine
} from "recharts";
import type { DailyForecast } from "../../api/types";
import { CHART_COLORS } from "../../utils/colours";
import { fmtDate } from "../../utils/units";

interface Props { daily: DailyForecast[]; }

export function HumidityChart({ daily }: Props) {
  const data = daily.map((d) => ({
    date: fmtDate(d.date),
    rh850: d.rh_850,
    rh500: d.rh_500,
  }));

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-secondary uppercase
                     tracking-wide mb-3">
        Relative Humidity (%) · 850 hPa bars · 500 hPa line
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data}
          margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3"
            stroke="#30363d" strokeOpacity={0.6} />
          <XAxis dataKey="date"
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <YAxis domain={[0, 110]} unit="%" width={40}
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <Tooltip
            contentStyle={{
              background: "#161b22", border: "1px solid #30363d",
              borderRadius: "8px", fontSize: "11px", color: "#e6edf3"
            }}
            formatter={(v: any) => v != null ? `${Math.round(v)}%` : "—"}
          />
          <Legend wrapperStyle={{ fontSize: "11px", color: "#8B949E" }} />
          <ReferenceLine y={80} stroke={CHART_COLORS.humidity850}
            strokeDasharray="4 4" strokeOpacity={0.5}
            label={{ value: "80% rain likely", fill: "#58A6FF", fontSize: 10 }} />
          <ReferenceLine y={30} stroke={CHART_COLORS.windMid}
            strokeDasharray="4 4" strokeOpacity={0.5}
            label={{ value: "30% dry", fill: "#EF9F27", fontSize: 10 }} />
          <Bar dataKey="rh850" name="RH 850 hPa"
            fill={CHART_COLORS.humidity850} opacity={0.75}
            radius={[3, 3, 0, 0]} maxBarSize={32} />
          <Line type="monotone" dataKey="rh500"
            name="RH 500 hPa"
            stroke={CHART_COLORS.humidity500} strokeWidth={2}
            dot={{ r: 3, fill: CHART_COLORS.humidity500 }}
            activeDot={{ r: 5 }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
