import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, Cell
} from "recharts";
import type { DailyForecast } from "../../api/types";
import { lapseRateColor } from "../../utils/colours";
import { fmtDate, stabilityLabel } from "../../utils/units";

interface Props { daily: DailyForecast[]; }

export function StabilityChart({ daily }: Props) {
  const data = daily.map((d) => ({
    date:      fmtDate(d.date),
    lapse:     d.lapse_rate,
    stability: d.stability ?? stabilityLabel(d.lapse_rate),
  }));

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-secondary uppercase
                     tracking-wide mb-3">
        Atmospheric Stability · Lapse Rate 850→500 hPa (°C/km)
      </h3>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data}
          margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3"
            stroke="#30363d" strokeOpacity={0.6} />
          <XAxis dataKey="date"
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <YAxis unit=" °C/km" width={56}
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <Tooltip
            contentStyle={{
              background: "#161b22", border: "1px solid #30363d",
              borderRadius: "8px", fontSize: "11px"
            }}
            formatter={(v: any, _: any, props: any) => [
              `${v?.toFixed(2)} °C/km — ${props.payload?.stability}`,
              "Lapse rate"
            ]}
          />
          <ReferenceLine y={9.8} stroke="#F78166" strokeDasharray="4 4"
            label={{ value: "9.8 DALR (unstable)", fill: "#F78166", fontSize: 10 }} />
          <ReferenceLine y={6.5} stroke="#3FB950" strokeDasharray="4 4"
            label={{ value: "6.5 standard", fill: "#3FB950", fontSize: 10 }} />
          <Bar dataKey="lapse" name="Lapse rate" maxBarSize={32}
            radius={[3, 3, 0, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={lapseRateColor(entry.lapse)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="flex gap-4 mt-2 text-xs text-text-secondary">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-[#3FB950] inline-block" />
          Stable (&lt;6.5)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-[#EF9F27] inline-block" />
          Conditional (6.5–9.8)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-[#F78166] inline-block" />
          Unstable (&gt;9.8)
        </span>
      </div>
    </div>
  );
}
