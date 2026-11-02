import { StrictMode, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  CalendarDays,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Download,
  ExternalLink,
  FileClock,
  Gauge,
  Home,
  Info,
  LayoutDashboard,
  Leaf,
  Lightbulb,
  LoaderCircle,
  Menu,
  Plus,
  RefreshCw,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Snowflake,
  Sparkles,
  Thermometer,
  Trash2,
  Users,
  Utensils,
  Waves,
  Upload,
  X,
  Zap,
} from "lucide-react";
import type {
  Appliance,
  Bootstrap,
  ReadingsReport,
  Run,
  SavedRun,
  Scenario,
} from "./types";
import "./styles.css";

type View = "overview" | "appliances" | "insights" | "history";
const views = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "appliances", label: "Appliances", icon: Home },
  { id: "insights", label: "Insights & rules", icon: Lightbulb },
  { id: "history", label: "Analysis history", icon: FileClock },
] as const;
const money = (n: number | null | undefined) =>
  n == null
    ? "Unavailable"
    : new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency: "INR",
        maximumFractionDigits: 2,
      }).format(n);
const date = (s: string) =>
  new Date(s.replace(" ", "T")).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
const clock = (s: string) => s.slice(11, 16);
const num = (n: number) => n.toFixed(2);
const statusLabel = (s: string) =>
  ({
    optimal: "Plan ready",
    infeasible: "Needs attention",
    search_limit: "Search limit reached",
    feasible_search_limit: "Feasible plan · review",
  })[s] || s;
async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message =
      "We couldn’t connect. Check that the local service is running and try again.";
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail))
        message = body.detail
          .map(
            (v: { loc: string[]; msg: string }) =>
              `${v.loc.slice(1).join(" › ")}: ${v.msg}`,
          )
          .join(". ");
    } catch {
      /* use connection message */
    }
    throw new Error(message);
  }
  return res.json();
}
function ApplianceIcon({ name, size = 21 }: { name: string; size?: number }) {
  const Icon = name.toLowerCase().includes("refriger")
    ? Snowflake
    : name.toLowerCase().includes("dish")
      ? Utensils
      : name.toLowerCase().includes("wash")
        ? Waves
        : name.toLowerCase().includes("air")
          ? Thermometer
          : Zap;
  return <Icon size={size} />;
}
function Pill({
  children,
  tone = "blue",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return <span className={`pill ${tone}`}>{children}</span>;
}
function PanelTitle({
  eyebrow,
  title,
  action,
}: {
  eyebrow?: string;
  title: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="panel-title">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}
function ScenarioField({
  label,
  field,
  min,
  max,
  step = 1,
  suffix,
  scenario,
  onChange,
}: {
  label: string;
  field: keyof Scenario;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  scenario: Scenario | null;
  onChange: (key: keyof Scenario, value: number) => void;
}) {
  return (
    <label className="field">
      <span>
        {label}
        {suffix && <small>{suffix}</small>}
      </span>
      <input
        required
        type="number"
        min={min}
        max={max}
        step={step}
        value={scenario?.[field] ?? ""}
        onChange={(e) => onChange(field, e.target.valueAsNumber)}
      />
    </label>
  );
}

export function DemandChart({ run }: { run: Run }) {
  const [hover, setHover] = useState<number | null>(null);
  const [range, setRange] = useState(24);
  const rows = run.history.slice(-range * 4);
  const f = run.forecast;
  const W = 760,
    H = 232,
    left = 42,
    right = 31,
    top = 24,
    bottom = 34;
  const start = new Date(
    (rows[0]?.time || f.origin).replace(" ", "T"),
  ).getTime();
  const end = new Date(f.target_end.replace(" ", "T")).getTime();
  const max =
    Math.ceil(
      Math.max(f.threshold_kw, f.forecast_kw, ...rows.map((r) => r.kw || 0)) *
        1.17 *
        2,
    ) / 2;
  const x = (time: string) =>
    left +
    ((new Date(time.replace(" ", "T")).getTime() - start) / (end - start)) *
      (W - left - right);
  const y = (v: number) => H - bottom - (v / max) * (H - top - bottom);
  const segments: string[] = [];
  let segment = "";
  for (const row of rows) {
    if (row.kw == null) {
      if (segment) segments.push(segment);
      segment = "";
    } else segment += `${segment ? "L" : "M"}${x(row.time)},${y(row.kw)} `;
  }
  if (segment) segments.push(segment);
  const selected = hover == null ? null : rows[hover];
  return (
    <section className="panel demand-panel">
      <PanelTitle
        title="Your electricity, a little clearer"
        action={
          <div className="segmented" aria-label="History range">
            {[6, 12, 24].map((h) => (
              <button
                key={h}
                aria-pressed={range === h}
                className={range === h ? "selected" : ""}
                onClick={() => setRange(h)}
              >
                {h}h
              </button>
            ))}
          </div>
        }
      />
      <div className="chart-meta">
        <span>
          <i className="dot blue-dot" />
          Recorded usage
        </span>
        <span>
          <i className="dot orange-dot" />
          Forecast interval
        </span>
        <span className="chart-unit">Average power · kW</span>
      </div>
      <div className="chart-wrap">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Electricity usage for the previous ${range} hours and forecast ${num(f.forecast_kw)} kW. Only the orange point is a forecast.`}
          onMouseLeave={() => setHover(null)}
        >
          <defs>
            <linearGradient id="area" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#2965e8" stopOpacity=".22" />
              <stop offset="100%" stopColor="#2965e8" stopOpacity="0" />
            </linearGradient>
          </defs>
          {[0, 1, 2, 3, 4].map((i) => (
            <g key={i}>
              <line
                x1={left}
                x2={W - right}
                y1={y((max * i) / 4)}
                y2={y((max * i) / 4)}
                stroke="#e8edf5"
                strokeDasharray="4 5"
              />
              <text
                x={left - 12}
                y={y((max * i) / 4) + 4}
                textAnchor="end"
                className="axis-text"
              >
                {((max * i) / 4).toFixed(1)}
              </text>
            </g>
          ))}
          <rect
            x={x(f.target_interval_start)}
            y={top}
            width={x(f.target_end) - x(f.target_interval_start)}
            height={H - bottom - top}
            fill="#fff0df"
          />
          <line
            x1={left}
            x2={W - right}
            y1={y(f.threshold_kw)}
            y2={y(f.threshold_kw)}
            stroke="#b4bfd3"
            strokeDasharray="5 5"
          />
          {segments.map((d, i) => (
            <path
              key={i}
              d={d}
              stroke="#2d63d4"
              strokeWidth="2.5"
              fill="none"
              strokeLinejoin="round"
            />
          ))}
          <circle
            cx={x(f.target_end)}
            cy={y(f.forecast_kw)}
            r="8"
            fill="#f78132"
            stroke="white"
            strokeWidth="3"
          />
          <text
            x={x(f.target_end) - 13}
            y={y(f.forecast_kw) - 14}
            textAnchor="end"
            fill="#a04b0c"
            fontSize="12"
            fontWeight="700"
          >
            {num(f.forecast_kw)} kW
          </text>
          {[0, 0.25, 0.5, 0.75, 1].map((p, i) => (
            <text
              key={i}
              x={left + p * (W - left - right)}
              y={H - 8}
              textAnchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}
              className="axis-text"
            >
              {new Date(start + p * (end - start)).toLocaleTimeString("en-GB", {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </text>
          ))}
          {rows.map((r, i) => (
            <rect
              key={r.time}
              x={x(r.time) - 3}
              y={top}
              width={Math.max(6, (W - left - right) / rows.length)}
              height={H - bottom - top}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
            />
          ))}
          {selected && selected.kw != null && (
            <g>
              <line
                x1={x(selected.time)}
                x2={x(selected.time)}
                y1={top}
                y2={H - bottom}
                stroke="#153c87"
                strokeDasharray="3 4"
              />
              <circle
                cx={x(selected.time)}
                cy={y(selected.kw)}
                r="4"
                fill="#153c87"
              />
            </g>
          )}
        </svg>
        {selected && (
          <div className="chart-tooltip">
            {clock(selected.time)} ·{" "}
            {selected.kw == null
              ? "Reading unavailable"
              : `${num(selected.kw)} kW`}
          </div>
        )}
      </div>
      <div className="chart-foot">
        <span>
          <Clock3 size={14} /> {date(f.origin)} ·{" "}
          {run.source_mode === "uploaded_readings"
            ? f.is_sample
              ? "synthetic sample readings"
              : "uploaded household readings"
            : "recorded household history"}
        </span>
        <span>
          Forecast: {date(f.target_end)} · {clock(f.target_interval_start)}–
          {clock(f.target_end)}
        </span>
      </div>
      <p className="microcopy">
        The forecast covers one 15-minute interval, one hour after the selected
        reading. It is not a full-day forecast.
      </p>
      <details className="data-details">
        <summary>View accessible readings table</summary>
        <div className="table-scroll compact-table">
          <table>
            <thead>
              <tr>
                <th>Recorded time</th>
                <th>Power (kW)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.time}>
                  <td>{r.time}</td>
                  <td>{r.kw == null ? "Unavailable" : num(r.kw)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}

export function ScheduleView({
  run,
  onAdjust,
}: {
  run: Run;
  onAdjust: () => void;
}) {
  const [mode, setMode] = useState("timeline");
  const f = run.forecast,
    p = run.plan;
  return (
    <section className="panel schedule-panel">
      <PanelTitle
        eyebrow="MAKE ROOM FOR WHAT MATTERS"
        title="A smarter appliance plan"
        action={
          <div className="segmented">
            {["timeline", "table"].map((m) => (
              <button
                key={m}
                aria-pressed={mode === m}
                className={mode === m ? "selected" : ""}
                onClick={() => setMode(m)}
              >
                {m === "timeline" ? "Timeline" : "Details"}
              </button>
            ))}
          </div>
        }
      />
      <div className="schedule-subtitle">
        <span>
          {clock(f.origin)} onwards · {run.scenario.horizon_hours}-hour plan
        </span>
        <Pill tone={p.status === "optimal" ? "green" : "orange"}>
          {statusLabel(p.status)}
        </Pill>
      </div>
      {p.schedule.length === 0 ? (
        <div className="empty compact">
          <CalendarDays />
          <h3>
            {p.status === "optimal"
              ? "No jobs need rescheduling"
              : "Let’s adjust this plan"}
          </h3>
          <p>{p.reason}</p>
          {p.status !== "optimal" && (
            <button className="secondary-button" onClick={onAdjust}>
              Review appliance timings <ArrowRight size={15} />
            </button>
          )}
        </div>
      ) : mode === "timeline" ? (
        <div className="timeline">
          <div className="timeline-head">
            <span>APPLIANCE</span>
            <div>
              {Array.from({ length: 5 }, (_, i) => (
                <span key={i}>
                  {clock(
                    new Date(
                      new Date(f.origin.replace(" ", "T")).getTime() +
                        (run.scenario.horizon_hours * 3600000 * i) / 4,
                    ).toLocaleString("sv-SE"),
                  )}
                </span>
              ))}
            </div>
          </div>
          {p.schedule.map((s, i) => {
            const offset =
              (new Date(s.Start.replace(" ", "T")).getTime() -
                new Date(f.origin.replace(" ", "T")).getTime()) /
              3600000;
            const duration =
              (new Date(s.End.replace(" ", "T")).getTime() -
                new Date(s.Start.replace(" ", "T")).getTime()) /
              3600000;
            return (
              <div key={s.Appliance} className="timeline-row">
                <span className="timeline-name">
                  <ApplianceIcon name={s.Appliance} size={17} />
                  {s.Appliance}
                </span>
                <div className="timeline-track">
                  <div
                    className={`timeline-bar bar-${i % 2}`}
                    style={{
                      left: `${(offset / run.scenario.horizon_hours) * 100}%`,
                      width: `${(duration / run.scenario.horizon_hours) * 100}%`,
                    }}
                    title={`${s.Appliance}: ${clock(s.Start)}–${clock(s.End)}, ${money(s.Estimated_cost)}`}
                  >
                    <span>{clock(s.Start)}</span>
                  </div>
                </div>
              </div>
            );
          })}
          <div className="timeline-legend">
            <span>
              <i className="dot blue-dot" />
              Recommended run time
            </span>
            <span>All times follow the selected history’s local clock</span>
          </div>
        </div>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Appliance</th>
                <th>Start → finish</th>
                <th>Energy</th>
                <th>Estimated cost</th>
              </tr>
            </thead>
            <tbody>
              {p.schedule.map((s) => (
                <tr key={s.Appliance}>
                  <td>{s.Appliance}</td>
                  <td>
                    {clock(s.Start)} → {clock(s.End)}
                    <small>
                      {date(s.Start)}
                      {s.Start.slice(0, 10) !== s.End.slice(0, 10)
                        ? ` → ${date(s.End)}`
                        : ""}
                    </small>
                  </td>
                  <td>{num(s.Energy_kWh)} kWh</td>
                  <td>{money(s.Estimated_cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="protected-strip">
        <ShieldCheck size={18} />
        <span>
          {run.recommendations
            .filter((r) => r.Kind === "protected")
            .map((r) => r.Appliance)
            .join(" & ") || "Household priorities"}
          <b> · protected by your household rules</b>
        </span>
      </div>
      <p className="microcopy">
        {p.reason} The parallel-job limit applies to scheduled jobs, not all
        household appliances.
      </p>
    </section>
  );
}

const friendlyFactor = (name: string) => {
  const s = name.toLowerCase().replaceAll("_", " ");
  if (s.includes("lstm pattern")) return "Learned usage patterns";
  if (s.includes("hour sin") || s.includes("hour cos")) return "Time of day";
  if (s.includes("dow sin") || s.includes("dow cos") || s.includes("weekend"))
    return "Day of the week";
  if (s.includes("month")) return "Time of year";
  if (s.includes("observed")) return "Availability of past readings";
  if (s.includes("previous week")) return "Usage at this time last week";
  if (s.includes("target slot previous day"))
    return "Usage at this time yesterday";
  if (s.includes("previous day")) return "Your previous day’s usage pattern";
  if (s.includes("mean previous")) return "Recent average electricity use";
  if (
    s.includes("before origin") ||
    s.includes("latest kw") ||
    s.includes("power scaled")
  )
    return "Recent electricity readings";
  return name.replace("Context: ", "").replaceAll("_", " ");
};
export function Explanations({ run }: { run: Run }) {
  const exp = run.explanation;
  const grouped = new Map<string, number>();
  for (const factor of exp.factors || []) {
    const name = friendlyFactor(factor.name);
    grouped.set(name, (grouped.get(name) || 0) + factor.value_kw);
  }
  const factors = Array.from(grouped, ([name, value_kw]) => ({
    name,
    value_kw,
  }))
    .sort((a, b) => Math.abs(b.value_kw) - Math.abs(a.value_kw))
    .slice(0, 8);
  const max = Math.max(...factors.map((f) => Math.abs(f.value_kw)), 0.01);
  return (
    <>
      <section className="panel">
        <PanelTitle
          eyebrow="UNDERSTAND YOUR FORECAST"
          title="What shaped this estimate"
          action={
            <Pill tone="neutral">
              {exp.available ? "Evidence available" : "Limited evidence"}
            </Pill>
          }
        />
        {exp.available ? (
          <>
            <p className="subtext">
              These are the strongest grouped contributions relative to the
              reference history. Orange raises the estimate; blue lowers it.
            </p>
            <div className="factor-list">
              {factors.map((f) => (
                <div className="factor" key={f.name}>
                  <div>
                    <span>{f.name}</span>
                    <b className={f.value_kw > 0 ? "warm-text" : "blue-text"}>
                      {f.value_kw > 0 ? "+" : ""}
                      {f.value_kw.toFixed(3)} kW
                    </b>
                  </div>
                  <div className="factor-track">
                    <span
                      className={f.value_kw > 0 ? "positive" : "negative"}
                      style={{
                        width: `${(Math.abs(f.value_kw) / max) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <div className="evidence-note">
              <Info size={17} />
              <p>{exp.note}</p>
            </div>
            <details className="evidence-details">
              <summary>How this explanation was checked</summary>
              <p>
                Method: {exp.method}. Reference forecast:{" "}
                {exp.reference_kw?.toFixed(3)} kW.
              </p>
              <p>
                Contribution reconstruction:{" "}
                {exp.reconstruction_verified
                  ? "within recorded tolerance"
                  : "outside recorded tolerance; interpret with caution"}
                .
                {exp.residual_kw != null &&
                  ` Residual: ${exp.residual_kw.toFixed(4)} kW.`}
              </p>
              <p>
                The full contribution list is included in the downloadable
                analysis.
              </p>
            </details>
          </>
        ) : (
          <div className="empty compact">
            <Lightbulb />
            <h3>No verified factors for this history</h3>
            <p>{exp.reason}</p>
            <p>
              Choose a history marked “with explanation” to inspect saved
              forecast contributions.
            </p>
          </div>
        )}
      </section>
      <section className="panel">
        <PanelTitle
          eyebrow="RULES WITH A REASON"
          title="Your household comes first"
        />
        <div className="rule-list">
          {run.recommendations.map((r) => (
            <details key={r.Appliance} className="rule-item">
              <summary>
                <span className={`appliance-icon ${r.Essential ? "mint" : ""}`}>
                  <ApplianceIcon name={r.Appliance} />
                </span>
                <span>
                  <strong>{r.Appliance}</strong>
                  <small>
                    {r.Action}
                    {r.Suggested_start ? ` · ${clock(r.Suggested_start)}` : ""}
                  </small>
                </span>
                <ChevronDown size={18} />
              </summary>
              <div className="rule-body">
                <p>{r.Reason}</p>
                <div className="rule-tags">
                  {r.Rule_trace.map((t) => (
                    <code key={t}>{t}</code>
                  ))}
                </div>
                <dl>
                  <dt>Recognized properties</dt>
                  <dd>{r.Ontology_types.join(", ")}</dd>
                  <dt>Imported rule candidates</dt>
                  <dd>{r.OWL_candidates.join(", ") || "None"}</dd>
                  <dt>Previously asserted examples</dt>
                  <dd>{r.OWL_asserted_decisions.join(", ") || "None"}</dd>
                </dl>
                <p className="microcopy">
                  Candidate decisions are resolved by the ordered household
                  rules. Essential protection takes precedence.
                </p>
              </div>
            </details>
          ))}
        </div>
      </section>
      {exp.activation && (
        <section className="panel">
          <PanelTitle title="Sequence response evidence" />
          <p className="subtext">
            Internal responses over the recorded history. These are not causal
            explanations or confidence scores.
          </p>
          <div className="activation-grid">
            {exp.activation.slice(0, 32).map((row, i) => (
              <div key={i}>
                {row.map((v, j) => (
                  <i
                    key={j}
                    title={`Unit ${i + 1}, step ${j + 1}: ${v.toFixed(3)}`}
                    style={{
                      background:
                        v > 0
                          ? `rgba(236,119,44,${Math.min(1, Math.abs(v))})`
                          : `rgba(37,91,207,${Math.min(1, Math.abs(v))})`,
                    }}
                  />
                ))}
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
