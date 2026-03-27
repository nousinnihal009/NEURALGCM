import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine
} from "recharts";
import type { DailyForecast } from "../../api/types";
import { CHART_COLORS } from "../../utils/colours";
import { fmtDate } from "../../utils/units";

interface Props { daily: DailyForecast[]; }

export function PressureChart({ daily }: Props) {
  const data = daily.map((d) => ({
    date: fmtDate(d.date),
    mslp: d.mslp_hpa,
  }));

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-secondary uppercase
                     tracking-wide mb-3">
        Mean Sea-Level Pressure (hPa) · falling=storm approaching
      </h3>
      <ResponsiveContainer width="100%" height={170}>
        <AreaChart data={data}
          margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="presGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#BD8EE6" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#BD8EE6" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3"
            stroke="#30363d" strokeOpacity={0.6} />
          <XAxis dataKey="date"
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <YAxis unit=" hPa" width={56}
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <Tooltip
            contentStyle={{
              background: "#161b22", border: "1px solid #30363d",
              borderRadius: "8px", fontSize: "11px"
            }}
            formatter={(v: any) => v != null ? `${v.toFixed(1)} hPa` : "—"}
          />
          <ReferenceLine y={1013.25} stroke="#e6edf3"
            strokeDasharray="4 4" strokeOpacity={0.3}
            label={{ value: "1013.25 standard", fill: "#8B949E", fontSize: 10 }} />
          <ReferenceLine y={1000} stroke="#F78166"
            strokeDasharray="4 4" strokeOpacity={0.4}
            label={{ value: "<1000 low pressure", fill: "#F78166", fontSize: 10 }} />
          <Area type="monotone" dataKey="mslp" name="MSLP"
            stroke={CHART_COLORS.pressure} strokeWidth={2}
            fill="url(#presGrad)"
            dot={{ r: 3, fill: CHART_COLORS.pressure }}
            activeDot={{ r: 5 }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
