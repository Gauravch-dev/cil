import { useState } from "react";
import {
  CheckCircle2,
  ChevronDown,
  Download,
  Info,
  LoaderCircle,
  Plus,
  ShieldCheck,
  Trash2,
  Upload,
  Zap,
} from "lucide-react";
import type { Appliance, ReadingsReport, Run, Scenario } from "./types";

export type Inputs = {
  scenario: Scenario;
  appliances: Appliance[];
  source: "uploaded" | "recorded" | "sample";
  caseId: string;
  report: ReadingsReport | null;
};
export type Evidence = {
  forecast_kw: number;
  model_sha256: string;
  prediction_reexecuted: boolean;
  history_shape: number[];
  context_shape: number[];
  history_channels: string[];
  context_features: { name: string; scaled_value: number }[];
  activation: number[][];
  comparison: { Model: string; MAE_kW: number; RMSE_kW: number; R2: number }[];
  classical_ml: {
    available: boolean;
    reason: string;
    model?: string;
    feature_count?: number;
    latent_features?: number;
    context_features?: number;
    latent_values?: { name: string; value: number }[];
    validation?: {
      Model: string;
      MAE_kW: number;
      RMSE_kW: number;
      R2: number;
    }[];
    counts?: { train: number; validation: number; test: number };
    run_id?: string;
  };
  shap: Run["explanation"] & {
    group_count?: number;
    permutations?: number;
    reference_training_index?: number;
    raw_forecast_kw?: number;
  };
};
export const clone = <T,>(x: T): T => structuredClone(x);
export const money = (n: number | null | undefined) =>
  n == null
    ? "Unavailable"
    : new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency: "INR",
        maximumFractionDigits: 2,
      }).format(n);
export const time = (s: string) => s.slice(11, 16);
export const day = (s: string) =>
  new Date(s.replace(" ", "T")).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
export const friendly = (s: string) =>
  s.replace("History: ", "History · ").replaceAll("_", " ");
export function downloadEvidence(value: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "current-demo-evidence.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export async function request<T>(
  url: string,
  options?: RequestInit,
): Promise<T> {
  const r = await fetch(url, options);
  const body = await r.json();
  if (!r.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : Array.isArray(body.detail)
          ? body.detail.map((v: { msg: string }) => v.msg).join(". ")
          : "The local service could not complete this request.",
    );
  return body;
}
export function Notice({
  children,
  tone = "blue",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return (
    <div className={`s-notice ${tone}`}>
      <Info size={18} />
      <div>{children}</div>
    </div>
  );
}
export function Stat({
  label,
  value,
  note,
}: {
  label: string;
  value: React.ReactNode;
  note?: string;
}) {
  return (
    <div className="s-stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}
function Field({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="s-field">
      <span>{label}</span>
      <input
        required
        type="number"
        min={min}
        max={max}
        step={step}
        value={Number.isFinite(value) ? value : ""}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
    </label>
  );
}
export function Preferences({
  value,
  onChange,
}: {
  value: Inputs;
  onChange: (v: Inputs) => void;
}) {
  const [tab, setTab] = useState("household");
  const set = (k: keyof Scenario, n: number) =>
    onChange({ ...value, scenario: { ...value.scenario, [k]: n } });
  const appliance = (
    i: number,
    k: keyof Appliance,
    n: string | number | boolean,
  ) =>
    onChange({
      ...value,
      appliances: value.appliances.map((a, j) =>
        j === i ? { ...a, [k]: n } : a,
      ),
    });
  return (
    <div className="s-preferences">
      <div className="s-tabs" role="tablist" aria-label="Preferences">
        {["household", "tariffs", "appliances"].map((x) => (
          <button
            type="button"
            key={x}
            role="tab"
            aria-selected={tab === x}
            onClick={() => setTab(x)}
          >
            {x}
          </button>
        ))}
      </div>
      <div role="tabpanel" aria-label={`${tab} preferences`}>
        {tab === "household" && (
          <>
            <div className="s-fields">
              <Field
                label="People at home"
                value={value.scenario.occupancy}
                max={20}
                onChange={(n) => set("occupancy", n)}
              />
              <Field
                label="Temperature (°C)"
                value={value.scenario.temperature_c}
                min={-50}
                max={60}
                onChange={(n) => set("temperature_c", n)}
              />
              <Field
                label="Planning horizon (hours)"
                value={value.scenario.horizon_hours}
                min={1}
                max={12}
                onChange={(n) => set("horizon_hours", n)}
              />
              <Field
                label="Parallel scheduled jobs"
                value={value.scenario.max_parallel_jobs}
                min={1}
                max={6}
                onChange={(n) => set("max_parallel_jobs", n)}
              />
            </div>
            <p className="s-help">
              These preferences guide the appliance plan. The demand estimate
              follows the electricity readings.
            </p>
          </>
        )}
        {tab === "tariffs" && (
          <>
            <div className="s-fields">
              <Field
                label="Off-peak price (₹/kWh)"
                value={value.scenario.offpeak_rate}
                max={10000}
                step={0.01}
                onChange={(n) => set("offpeak_rate", n)}
              />
              <Field
                label="Peak price (₹/kWh)"
                value={value.scenario.peak_rate}
                max={10000}
                step={0.01}
                onChange={(n) => set("peak_rate", n)}
              />
              <Field
                label="Peak starts (hour, 0–23)"
                value={value.scenario.peak_start_hour}
                max={23}
                onChange={(n) => set("peak_start_hour", n)}
              />
              <Field
                label="Peak ends (hour, 0–23)"
                value={value.scenario.peak_end_hour}
                max={23}
                onChange={(n) => set("peak_end_hour", n)}
              />
            </div>
            <p className="s-help">
              Use the same clock as your readings. These are supplied prices,
              not a connected utility tariff.
            </p>
          </>
        )}
        {tab === "appliances" && (
          <div className="s-appliance-list">
            {value.appliances.map((a, i) => (
              <details key={i} className="s-appliance">
                <summary>
                  <span className="s-small-icon">
                    <Zap size={17} />
                  </span>
                  <span>
                    <strong>{a.name}</strong>
                    <small>
                      {a.power_w} W · {a.duration_min} min ·{" "}
                      {a.essential
                        ? "Essential"
                        : a.flexible
                          ? "Flexible"
                          : "Fixed timing"}
                    </small>
                  </span>
                  <ChevronDown size={16} />
                </summary>
                <div className="s-appliance-body">
                  <label className="s-field">
                    <span>Appliance name</span>
                    <input
                      required
                      value={a.name}
                      maxLength={80}
                      onChange={(e) => appliance(i, "name", e.target.value)}
                    />
                  </label>
                  <div className="s-fields">
                    <Field
                      label="Power (W)"
                      value={a.power_w}
                      min={1}
                      max={20000}
                      onChange={(n) => appliance(i, "power_w", n)}
                    />
                    <Field
                      label="Runtime (minutes)"
                      value={a.duration_min}
                      min={15}
                      max={720}
                      step={15}
                      onChange={(n) => appliance(i, "duration_min", n)}
                    />
                    <Field
                      label="Earliest start (minutes)"
                      value={a.earliest_min}
                      max={720}
                      step={15}
                      onChange={(n) => appliance(i, "earliest_min", n)}
                    />
                    <Field
                      label="Finish within (minutes)"
                      value={a.deadline_min}
                      min={15}
                      max={720}
                      step={15}
                      onChange={(n) => appliance(i, "deadline_min", n)}
                    />
                  </div>
                  <label className="s-field">
                    <span>Priority</span>
                    <select
                      value={a.priority}
                      onChange={(e) => appliance(i, "priority", e.target.value)}
                    >
                      {["Low", "Medium", "High"].map((x) => (
                        <option key={x}>{x}</option>
                      ))}
                    </select>
                  </label>
                  <div className="s-checks">
                    {(
                      [
                        ["requested", "Include this appliance"],
                        ["on", "Already running"],
                        ["essential", "Essential appliance"],
                        ["flexible", "Allow rescheduling"],
                        ["requires_occupancy", "Needs someone home"],
                      ] as const
                    ).map(([key, label]) => (
                      <label key={key}>
                        <input
                          type="checkbox"
                          checked={a[key]}
                          disabled={
                            key === "essential" &&
                            a.ontology_id === "Refrigerator_1"
                          }
                          onChange={(e) => appliance(i, key, e.target.checked)}
                        />
                        {label}
                      </label>
                    ))}
                  </div>
                  {i > 0 && (
                    <button
                      type="button"
                      className="s-text danger"
                      onClick={() =>
                        onChange({
                          ...value,
                          appliances: value.appliances.filter(
                            (_, j) => j !== i,
                          ),
                        })
                      }
                    >
                      <Trash2 size={14} />
                      Remove appliance
                    </button>
                  )}
                </div>
              </details>
            ))}
            <button
              type="button"
              className="s-button secondary"
              disabled={value.appliances.length >= 8}
              onClick={() =>
                onChange({
                  ...value,
                  appliances: [
                    ...value.appliances,
                    {
                      name: `Appliance ${value.appliances.length + 1}`,
                      power_w: 500,
                      duration_min: 60,
                      requested: true,
                      on: false,
                      essential: false,
                      flexible: true,
                      priority: "Medium",
                      earliest_min: 0,
                      deadline_min: 360,
                      requires_occupancy: false,
                      ontology_id: "",
                    },
                  ],
                })
              }
            >
              <Plus size={16} />
              Add appliance <small>{value.appliances.length}/8</small>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
export function MeterUpload({
  value,
  onChange,
  onBusy,
  onError,
  demoMode = false,
}: {
  value: Inputs;
  onChange: (v: Inputs) => void;
  onBusy: (v: boolean) => void;
  onError: (s: string) => void;
  demoMode?: boolean;
}) {
  const [file, setFile] = useState<File | null>(null),
    [checking, setChecking] = useState(false),
    [zone, setZone] = useState("Asia/Kolkata");
  async function check() {
    if (!file) return;
    setChecking(true);
    onBusy(true);
    onError("");
    try {
      const report = await request<ReadingsReport>(
        `/api/readings?timezone=${encodeURIComponent(zone)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "text/csv",
            "X-File-Name": encodeURIComponent(file.name),
          },
          body: file,
        },
      );
      onChange({ ...value, report });
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setChecking(false);
      onBusy(false);
    }
  }
  return (
    <div className="s-upload">
      <label className="s-drop">
        <span className="s-upload-icon">
          <Upload size={24} />
        </span>
        <strong>{file?.name || "Choose your meter readings"}</strong>
        <span>CSV · at least 8 days · up to 10 MB</span>
        <input
          type="file"
          aria-label="Choose electricity readings CSV"
          accept=".csv,text/csv"
          disabled={checking}
          onChange={(e) => {
            setFile(e.target.files?.[0] || null);
            onChange({ ...value, report: null });
          }}
        />
      </label>
      <div className="s-upload-actions">
        <label className="s-field">
          <span>Reading timezone</span>
          <select
            value={zone}
            disabled={checking}
            onChange={(e) => {
              setZone(e.target.value);
              onChange({ ...value, report: null });
            }}
          >
            {[
              "Asia/Kolkata",
              "UTC",
              "Europe/Paris",
              "Europe/London",
              "America/New_York",
              "America/Los_Angeles",
              "Asia/Singapore",
              "Australia/Sydney",
            ].map((z) => (
              <option key={z}>{z}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="s-button secondary"
          disabled={!file || checking}
          onClick={() => void check()}
        >
          {checking ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <ShieldCheck size={16} />
          )}
          Check readings
        </button>
      </div>
      <div className="s-template-links">
        <a href="/api/readings/template" download>
          <Download size={14} />
          Blank CSV
        </a>
        <a href="/api/readings/samples/guide" download>
          Column & units guide
        </a>
      </div>
      {value.report && (
        <div className="s-upload-report">
          <div className="s-inline-between">
            <strong>
              {value.report.can_analyze
                ? "Readings ready"
                : demoMode && value.report.sample_ready
                  ? "Sample ready"
                  : "Review your readings"}
            </strong>
            <span className="s-tag">
              {value.report.row_count.toLocaleString()} rows
            </span>
          </div>
          <p>
            {day(value.report.start)} → {day(value.report.end)} ·{" "}
            {value.report.timezone}
          </p>
          {!value.report.can_analyze &&
            !(demoMode && value.report.sample_ready) && (
              <Notice tone="orange">{value.report.blockers[0]}</Notice>
            )}
          <details>
            <summary>View file checks and preview</summary>
            <div className="s-check-report">
              {value.report.checks.map((c) => (
                <p key={c.code}>
                  <span
                    className={c.status === "pass" ? "s-green" : "s-orange"}
                  >
                    {c.status === "pass" ? (
                      <CheckCircle2 size={15} />
                    ) : (
                      <Info size={15} />
                    )}
                  </span>
                  <span>
                    <b>{c.label}</b>
                    <small>{c.message}</small>
                  </span>
                </p>
              ))}
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Active power</th>
                    <th>Voltage</th>
                  </tr>
                </thead>
                <tbody>
                  {value.report.preview.map((r, i) => (
                    <tr key={i}>
                      <td>{r.Datetime}</td>
                      <td>{r.Global_active_power} kW</td>
                      <td>{r.Voltage} V</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {value.report.warnings.map((w) => (
              <p className="s-help" key={w}>
                {w}
              </p>
            ))}
          </details>
        </div>
      )}
      <details className="s-format">
        <summary>What readings do I need?</summary>
        <p>
          One row per minute with Datetime, Global_active_power,
          Global_reactive_power, Voltage, Global_intensity, Sub_metering_1,
          Sub_metering_2 and Sub_metering_3. Use the template’s units. A current
          estimate needs the latest complete 15-minute interval within 30
          minutes.
        </p>
        <p>
          Monthly bills and cumulative meter totals do not match this format.
          Files are stored locally. Clearing a selection does not delete saved
          uploads.
        </p>
      </details>
    </div>
  );
}
