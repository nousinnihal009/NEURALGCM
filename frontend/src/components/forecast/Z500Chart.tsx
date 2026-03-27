import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine
} from "recharts";
import type { DailyForecast } from "../../api/types";
import { CHART_COLORS } from "../../utils/colours";
import { fmtDate } from "../../utils/units";

interface Props { daily: DailyForecast[]; }

export function Z500Chart({ daily }: Props) {
  const data = daily.map((d) => ({
    date: fmtDate(d.date),
    z500: d.z500_m,
  }));

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-secondary uppercase
                     tracking-wide mb-3">
        Geopotential Height Z500 (m) · primary synoptic weather indicator
      </h3>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={data}
          margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3"
            stroke="#30363d" strokeOpacity={0.6} />
          <XAxis dataKey="date"
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <YAxis unit=" m" width={56}
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <Tooltip
            contentStyle={{
              background: "#161b22", border: "1px solid #30363d",
              borderRadius: "8px", fontSize: "11px"
            }}
            formatter={(v: any) => v != null ? `${Math.round(v)} m` : "—"}
          />
          <ReferenceLine y={5500} stroke={CHART_COLORS.z500}
            strokeDasharray="4 4" strokeOpacity={0.4}
            label={{ value: "5500m tropical threshold", fill: "#BD8EE6", fontSize: 10 }} />
          <Line type="monotone" dataKey="z500" name="Z500"
            stroke={CHART_COLORS.z500} strokeWidth={2.5}
            dot={{ r: 4, fill: CHART_COLORS.z500 }}
            activeDot={{ r: 6 }} />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-text-muted mt-1">
        LOW Z500 = troughs / storms · HIGH Z500 = ridges / fair weather
      </p>
    </div>
  );
}
