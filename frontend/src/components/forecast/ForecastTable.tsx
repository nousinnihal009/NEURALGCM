import type { ForecastResult } from "../../api/types";
import {
  fmtTemp, fmtRH, fmtWind, fmtTPW,
  fmtZ500, fmtPressure, fmtDate, stabilityLabel, stabilityColor
} from "../../utils/units";

interface Props { forecast: ForecastResult; }

export function ForecastTable({ forecast }: Props) {
  return (
    <div className="overflow-x-auto mt-4">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-border">
            {["Date","T°C","RH%","Wind","TPW","Z500","SP","Stability"].map(h => (
              <th key={h}
                className="text-left py-2 px-2 text-text-secondary
                           font-medium whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {forecast.daily.map((d, i) => {
            const lr = d.lapse_rate;
            return (
              <tr key={d.date}
                className={`border-b border-border/50 hover:bg-bg-tertiary
                            transition-colors ${i % 2 === 0 ? "" : "bg-bg-secondary/30"}`}>
                <td className="py-2 px-2 font-medium text-text-primary whitespace-nowrap">
                  {fmtDate(d.date)}
                </td>
                <td className="py-2 px-2 text-[#F78166]">
                  {fmtTemp(d.temperature_c_850)}
                </td>
                <td className="py-2 px-2 text-[#58A6FF]">
                  {fmtRH(d.rh_850)}
                </td>
                <td className="py-2 px-2 text-[#3FB950] whitespace-nowrap">
                  {fmtWind(d.wind_speed_850)}{" "}
                  <span className="text-text-muted">{d.wind_dir_compass}</span>
                </td>
                <td className="py-2 px-2 text-[#A5D6FF]">
                  {fmtTPW(d.tpw_mm)}
                </td>
                <td className="py-2 px-2 text-[#BD8EE6]">
                  {fmtZ500(d.z500_m)}
                </td>
                <td className="py-2 px-2 text-[#BD8EE6]">
                  {fmtPressure(d.mslp_hpa)}
                </td>
                <td className="py-2 px-2 whitespace-nowrap font-medium"
                  style={{ color: stabilityColor(lr) }}>
                  {stabilityLabel(lr)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
