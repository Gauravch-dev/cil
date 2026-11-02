export type Appliance = {
  name: string;
  power_w: number;
  duration_min: number;
  requested: boolean;
  on: boolean;
  essential: boolean;
  flexible: boolean;
  priority: string;
  earliest_min: number;
  deadline_min: number;
  requires_occupancy: boolean;
  ontology_id: string;
};
export type Scenario = {
  occupancy: number;
  temperature_c: number;
  peak_start_hour: number;
  peak_end_hour: number;
  peak_rate: number;
  offpeak_rate: number;
  horizon_hours: number;
  max_parallel_jobs: number;
  node_limit: number;
};
export type Bootstrap = {
  mode: string;
  cases: { id: string; origin: string; explanation: boolean }[];
  default_case: string;
  scenario: Scenario;
  appliances: Appliance[];
  integration_gap: string | null;
  readings: ReadingsSchema & { service_available: boolean };
  ontology: {
    classes: number;
    rules: { id: string; classes: string[]; decisions: string[] }[];
  };
};
export type Forecast = {
  origin: string;
  target_end: string;
  target_interval_start: string;
  forecast_kw: number;
  threshold_kw: number;
  model: string;
  input_scope?: string;
  is_sample?: boolean;
};
export type Schedule = {
  Appliance: string;
  Start: string;
  End: string;
  Power_W: number;
  Energy_kWh: number;
  Estimated_cost: number;
};
export type Plan = {
  status: string;
  reason: string;
  schedule: Schedule[];
  estimated_cost: number | null;
  baseline_cost: number | null;
  estimated_saving: number | null;
  nodes: number;
};
export type Recommendation = {
  Appliance: string;
  Action: string;
  Reason: string;
  Kind: string;
  Essential: boolean;
  Flexible: boolean;
  Rule_trace: string[];
  OWL_candidates: string[];
  OWL_asserted_decisions: string[];
  Ontology_types: string[];
  Suggested_start?: string;
};
export type Run = {
  id: string;
  created_at: string;
  case_id: string;
  source_mode: string;
  upload_id?: string;
  uploaded_readings?: ReadingsReport;
  forecast: Forecast;
  scenario: Scenario;
  appliances: Appliance[];
  recommendations: Recommendation[];
  plan: Plan;
  history: { time: string; kw: number | null }[];
  confidence: { available: boolean; label: string; reason: string };
  explanation: {
    available: boolean;
    reason?: string;
    method?: string;
    reference_kw?: number;
    reconstruction_verified?: boolean;
    residual_kw?: number;
    note?: string;
    activation?: number[][];
    factors?: { name: string; value_kw: number }[];
  };
  ontology_sha256: string;
};
export type ReadingsSchema = {
  columns: { name: string; unit: string }[];
  minimum_days: number;
  maximum_days: number;
  maximum_bytes: number;
  maximum_age_minutes: number;
  template_url: string;
};
export type ReadingsReport = {
  id: string;
  file_name: string;
  byte_size: number;
  sha256: string;
  timezone: string;
  row_count: number;
  start: string;
  end: string;
  days_covered: number;
  origin: string | null;
  target_interval_start: string | null;
  target_end: string | null;
  age_minutes: number | null;
  invalid_cells: number;
  is_sample?: boolean;
  sample_ready?: boolean;
  missing_minutes: number;
  data_ready: boolean;
  forecaster_available: boolean;
  can_analyze: boolean;
  status: "ready" | "needs_service" | "needs_readings";
  checked_at: string;
  checks: {
    code: string;
    label: string;
    status: "pass" | "warn" | "fail";
    message: string;
  }[];
  blockers: string[];
  warnings: string[];
  preview: Record<string, string | number | null>[];
  history: { time: string; kw: number | null }[];
};
export type SavedRun = Pick<
  Run,
  "id" | "created_at" | "case_id" | "source_mode" | "forecast" | "plan"
>;
