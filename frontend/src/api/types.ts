export type ForecastMode = "realtime" | "historical";
export type ForecastStatus =
  "pending" | "running" | "complete" | "failed" | "cached";

export interface ForecastRequest {
  location_name: string;
  lat: number;
  lon: number;
  days?: number;
  mode?: ForecastMode;
  init_date?: string;
}

export interface DailyForecast {
  date: string;
  temperature_c_850:     number | null;
  temperature_c_500:     number | null;
  rh_850:                number | null;
  rh_500:                number | null;
  specific_humidity_850: number | null;
  tpw_mm:                number | null;
  wind_speed_850:        number | null;
  wind_speed_500:        number | null;
  wind_speed_250:        number | null;
  wind_dir_850:          number | null;
  wind_dir_compass:      string | null;
  u_850:                 number | null;
  v_850:                 number | null;
  z500_m:                number | null;
  mslp_hpa:              number | null;
  lapse_rate:            number | null;
  stability:             string | null;
  clwc_gkg_850:          number | null;
  ciwc_gkg_850:          number | null;
  vorticity_850:         number | null;
}

export interface ForecastResult {
  job_id:             string;
  status:             ForecastStatus;
  location_name:      string;
  lat:                number;
  lon:                number;
  model_lat:          number | null;
  model_lon:          number | null;
  init_time_utc:      string | null;
  mode_used:          string | null;
  forecast_days:      number;
  elapsed_seconds:    number | null;
  is_cached:          boolean;
  created_at:         string;
  daily:              DailyForecast[];
  sanity_ok:          boolean | null;
  sanity_violations:  string[] | null;
  png_url:            string | null;
  csv_url:            string | null;
  error:              string | null;
  model_checkpoint:   string;
  paper_reference:    string;
}

export interface ForecastJobResponse {
  job_id:             string;
  status:             ForecastStatus;
  message:            string;
  poll_url:           string;
  estimated_seconds:  number;
}

export interface MapLocation {
  lat: number;
  lon: number;
  name: string;
}
