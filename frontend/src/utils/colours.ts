export const CHART_COLORS = {
  temperature850: "#F78166",
  temperature500: "#EF9F27",
  era5Truth:      "#e6edf3",
  humidity850:    "#58A6FF",
  humidity500:    "#1D9E75",
  windSurface:    "#3FB950",
  windMid:        "#EF9F27",
  windJet:        "#F78166",
  windU:          "#EF9F27",
  windV:          "#D85A30",
  z500:           "#BD8EE6",
  pressure:       "#BD8EE6",
  tpw:            "#A5D6FF",
  stableBar:      "#3FB950",
  unstableBar:    "#F78166",
  conditionalBar: "#EF9F27",
  vorticityCyc:   "#F78166",
  vorticityAnti:  "#58A6FF",
  uncertainty:    "rgba(247,129,102,0.12)",
};

export function lapseRateColor(lr: number | null): string {
  if (lr === null) return CHART_COLORS.stableBar;
  if (lr > 9.8)   return CHART_COLORS.unstableBar;
  if (lr > 7.0)   return CHART_COLORS.conditionalBar;
  return CHART_COLORS.stableBar;
}
