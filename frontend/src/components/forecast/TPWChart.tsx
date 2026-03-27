import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine
} from "recharts";
import type { DailyForecast } from "../../api/types";
import { CHART_COLORS } from "../../utils/colours";
import { fmtDate } from "../../utils/units";

interface Props { daily: DailyForecast[]; }

export function TPWChart({ daily }: Props) {
  const data = daily.map((d) => ({
    date: fmtDate(d.date),
    tpw:  d.tpw_mm,
  }));

  return (
    <div>
      <h3 className="text-xs font-semibold text-text-secondary uppercase
                     tracking-wide mb-3">
        Total Precipitable Water (mm) · vertical integral of humidity
      </h3>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={data}
          margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="tpwGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"   stopColor="#A5D6FF" stopOpacity={0.4} />
              <stop offset="95%"  stopColor="#A5D6FF" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3"
            stroke="#30363d" strokeOpacity={0.6} />
          <XAxis dataKey="date"
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <YAxis unit=" mm" width={48}
            tick={{ fill: "#8B949E", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }} />
          <Tooltip
            contentStyle={{
              background: "#161b22", border: "1px solid #30363d",
              borderRadius: "8px", fontSize: "11px"
            }}
            formatter={(v: any) => v != null ? `${v.toFixed(1)} mm` : "—"}
          />
          <ReferenceLine y={60} stroke="#F78166" strokeDasharray="4 4"
            strokeOpacity={0.5}
            label={{ value: ">60mm heavy rain risk", fill: "#F78166", fontSize: 10 }} />
          <ReferenceLine y={20} stroke="#EF9F27" strokeDasharray="4 4"
            strokeOpacity={0.5}
            label={{ value: "<20mm dry", fill: "#EF9F27", fontSize: 10 }} />
          <Area type="monotone" dataKey="tpw" name="TPW"
            stroke={CHART_COLORS.tpw} strokeWidth={2.5}
            fill="url(#tpwGrad)"
            dot={{ r: 4, fill: CHART_COLORS.tpw }}
            activeDot={{ r: 6 }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
