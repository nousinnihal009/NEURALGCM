export function fmtTemp(v: number | null): string {
  if (v === null) return "—";
  return `${v.toFixed(1)}°C`;
}
export function fmtRH(v: number | null): string {
  if (v === null) return "—";
  return `${Math.round(v)}%`;
}
export function fmtWind(v: number | null): string {
  if (v === null) return "—";
  return `${v.toFixed(1)} m/s`;
}
export function fmtTPW(v: number | null): string {
  if (v === null) return "—";
  return `${v.toFixed(1)} mm`;
}
export function fmtZ500(v: number | null): string {
  if (v === null) return "—";
  return `${Math.round(v)} m`;
}
export function fmtPressure(v: number | null): string {
  if (v === null) return "—";
  return `${v.toFixed(1)} hPa`;
}
export function fmtLapse(v: number | null): string {
  if (v === null) return "—";
  return `${v.toFixed(2)} °C/km`;
}
export function fmtDate(dateStr: string): string {
  const d = new Date(dateStr + "T00:00:00Z");
  return d.toLocaleDateString("en-GB", {
    weekday: "short", day: "numeric", month: "short",
    timeZone: "UTC",
  });
}
export function stabilityLabel(lr: number | null): string {
  if (lr === null) return "Unknown";
  if (lr > 9.8)   return "UNSTABLE";
  if (lr > 7.0)   return "Cond. unstable";
  return "Stable";
}
export function stabilityColor(lr: number | null): string {
  if (lr === null) return "#8B949E";
  if (lr > 9.8)   return "#F78166";
  if (lr > 7.0)   return "#EF9F27";
  return "#3FB950";
}
