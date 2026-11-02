import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
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
import ReadingsUpload from "./ReadingsUpload";
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

function DemandChart({ run }: { run: Run }) {
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
            ? f.is_sample ? "synthetic sample readings" : "uploaded household readings"
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

function ScheduleView({ run, onAdjust }: { run: Run; onAdjust: () => void }) {
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
function Explanations({ run }: { run: Run }) {
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

function App() {
  const [templatesOpen, setTemplatesOpen] = useState(false);
  const [dataMode, setDataMode] = useState<"recorded" | "uploaded">("recorded");
  const [readingsReport, setReadingsReport] = useState<ReadingsReport | null>(
    null,
  );
  const [uploading, setUploading] = useState(false);
  const [view, setView] = useState<View>("overview"),
    [boot, setBoot] = useState<Bootstrap | null>(null),
    [run, setRun] = useState<Run | null>(null),
    [saved, setSaved] = useState<SavedRun[]>([]);
  const [scenario, setScenario] = useState<Scenario | null>(null),
    [appliances, setAppliances] = useState<Appliance[]>([]),
    [caseId, setCaseId] = useState("");
  const [busy, setBusy] = useState(false),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [dirty, setDirty] = useState(false),
    [drawer, setDrawer] = useState(false),
    [mobileMenu, setMobileMenu] = useState(false),
    [search, setSearch] = useState(""),
    [historyFilter, setHistoryFilter] = useState("all");
  const [drawerTab, setDrawerTab] = useState("household");
  const started = useRef(false);
  const closeRef = useRef<HTMLButtonElement>(null);
  const openRef = useRef<HTMLElement | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const navRef = useRef<HTMLElement>(null);
  const navCloseRef = useRef<HTMLButtonElement>(null);
  const resultRef = useRef<HTMLElement>(null);
  async function load() {
    setLoading(true);
    setError("");
    try {
      const [b, h] = await Promise.all([
        api<Bootstrap>("/api/bootstrap"),
        api<SavedRun[]>("/api/history"),
      ]);
      setBoot(b);
      setSaved(h);
      setScenario(b.scenario);
      setAppliances(b.appliances);
      setCaseId(b.default_case);
      if (h.length) {
        const r = await api<Run>(`/api/history/${h[0].id}`);
        setRun(r);
        setScenario(r.scenario);
        setAppliances(r.appliances);
        if (r.source_mode === "uploaded_readings" && r.upload_id) {
          setDataMode("uploaded");
          setReadingsReport(
            await api<ReadingsReport>(`/api/readings/${r.upload_id}`),
          );
        } else {
          setCaseId(r.case_id);
          setDataMode("recorded");
        }
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    if (!started.current) {
      started.current = true;
      void load();
    }
  }, []);
  useEffect(() => {
    if (!drawer) return;
    openRef.current = document.activeElement as HTMLElement;
    closeRef.current?.focus();
    const old = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = old;
      openRef.current?.focus();
    };
  }, [drawer]);
  useEffect(() => {
    if (!mobileMenu) return;
    const previous = document.activeElement as HTMLElement;
    const old = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    navCloseRef.current?.focus();
    return () => {
      document.body.style.overflow = old;
      previous?.focus();
    };
  }, [mobileMenu]);
  function updateScenario(key: keyof Scenario, value: number) {
    setScenario((s) => (s ? { ...s, [key]: value } : s));
    setDirty(true);
  }
  function updateAppliance(
    index: number,
    key: keyof Appliance,
    value: string | number | boolean,
  ) {
    setAppliances((a) =>
      a.map((item, i) => (i === index ? { ...item, [key]: value } : item)),
    );
    setDirty(true);
  }
  async function analyze() {
    if (!scenario) return;
    if (dataMode === "uploaded" && !readingsReport?.can_analyze && !readingsReport?.sample_ready) return;
    setBusy(true);
    setError("");
    try {
      const r = await api<Run>("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...(dataMode === "uploaded"
            ? { upload_id: readingsReport!.id, sample_forecast: !!readingsReport?.sample_ready }
            : { case_id: caseId }),
          scenario,
          appliances,
        }),
      });
      setRun(r);
      if (r.uploaded_readings) setReadingsReport(r.uploaded_readings);
      setDirty(false);
      setDrawer(false);
      setSaved(await api<SavedRun[]>("/api/history"));
      setView("overview");
      setTimeout(() => resultRef.current?.focus(), 0);
    } catch (e) {
      setError((e as Error).message);
      if (dataMode === "uploaded" && readingsReport) {
        try {
          setReadingsReport(
            await api<ReadingsReport>(`/api/readings/${readingsReport.id}`),
          );
        } catch {
          /* Preserve the original actionable error. */
        }
      }
    } finally {
      setBusy(false);
    }
  }
  async function restore(id: string) {
    setError("");
    setLoading(true);
    try {
      const r = await api<Run>(`/api/history/${id}`);
      setRun(r);
      setScenario(r.scenario);
      setAppliances(r.appliances);
      if (r.source_mode === "uploaded_readings" && r.upload_id) {
        setDataMode("uploaded");
        setReadingsReport(
          await api<ReadingsReport>(`/api/readings/${r.upload_id}`),
        );
      } else {
        setCaseId(r.case_id);
        setDataMode("recorded");
      }
      setDirty(false);
      setView("overview");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  const f = run?.forecast,
    p = run?.plan;
  const high = f ? f.forecast_kw >= f.threshold_kw : false;
  const selectedCase = boot?.cases.find((c) => c.id === caseId);
  const inputOrigin =
    dataMode === "uploaded" ? readingsReport?.origin : selectedCase?.origin;
  const resultMatchesSource =
    !!run &&
    (dataMode === "uploaded"
      ? run.source_mode === "uploaded_readings" &&
        run.upload_id === readingsReport?.id
      : run.source_mode !== "uploaded_readings");
  const canRun =
    !!boot &&
    !!scenario &&
    !busy &&
    !loading &&
    !uploading &&
    (dataMode === "recorded" || !!readingsReport?.can_analyze || !!readingsReport?.sample_ready);
  const navigate = (v: View) => {
    setView(v);
    setMobileMenu(false);
    setError("");
  };
  const applianceEditor = (
    <div className="appliance-editor">
      {appliances.map((a, i) => (
        <details
          key={i}
          className="appliance-edit"
          open={appliances.length === 1 || undefined}
        >
          <summary>
            <span className="appliance-icon">
              <ApplianceIcon name={a.name} />
            </span>
            <span>
              <strong>{a.name}</strong>
              <small>
                {a.power_w} W · {a.duration_min} min ·{" "}
                {a.essential || a.ontology_id === "Refrigerator_1"
                  ? "Essential"
                  : a.flexible
                    ? "Flexible"
                    : "Fixed timing"}
              </small>
            </span>
            <ChevronDown size={18} />
          </summary>
          <div className="appliance-fields">
            <div className="form-grid">
              <label className="field">
                <span>Name</span>
                <input
                  required
                  maxLength={80}
                  value={a.name}
                  onChange={(e) => updateAppliance(i, "name", e.target.value)}
                />
              </label>
              <label className="field">
                <span>
                  Power <small>W</small>
                </span>
                <input
                  required
                  type="number"
                  min="1"
                  max="20000"
                  value={a.power_w}
                  onChange={(e) =>
                    updateAppliance(i, "power_w", e.target.valueAsNumber)
                  }
                />
              </label>
              {(["duration_min", "earliest_min", "deadline_min"] as const).map(
                (key, j) => (
                  <label className="field" key={key}>
                    <span>
                      {["Run duration", "Earliest start", "Finish within"][j]}
                      <small>minutes</small>
                    </span>
                    <input
                      required
                      type="number"
                      min={key === "earliest_min" ? 0 : 15}
                      max="720"
                      step="15"
                      value={a[key]}
                      onChange={(e) =>
                        updateAppliance(i, key, e.target.valueAsNumber)
                      }
                    />
                  </label>
                ),
              )}
              <label className="field">
                <span>Priority</span>
                <select
                  value={a.priority}
                  onChange={(e) =>
                    updateAppliance(i, "priority", e.target.value)
                  }
                >
                  {["Low", "Medium", "High"].map((k) => (
                    <option key={k}>{k}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className="check-grid">
              {(
                [
                  "requested",
                  "on",
                  "essential",
                  "flexible",
                  "requires_occupancy",
                ] as const
              ).map((key, j) => (
                <label key={key} className="check-field">
                  <input
                    type="checkbox"
                    checked={
                      key === "essential" && a.ontology_id === "Refrigerator_1"
                        ? true
                        : a[key]
                    }
                    disabled={
                      key === "essential" && a.ontology_id === "Refrigerator_1"
                    }
                    onChange={(e) => updateAppliance(i, key, e.target.checked)}
                  />
                  {
                    [
                      "Include this appliance",
                      "Already running",
                      "Essential appliance",
                      "Allow rescheduling",
                      "Needs someone home",
                    ][j]
                  }
                </label>
              ))}
            </div>
            <p className="microcopy">
              Start and finish limits are relative to{" "}
              {inputOrigin ? clock(inputOrigin) : "the selected reading"}. Use
              15-minute steps. Essential appliances remain protected.
            </p>
            {a.ontology_id !== "Refrigerator_1" && (
              <button
                className="text-button danger"
                type="button"
                onClick={() => {
                  setAppliances((list) => list.filter((_, j) => j !== i));
                  setDirty(true);
                }}
              >
                <Trash2 size={14} />
                Remove appliance
              </button>
            )}
          </div>
        </details>
      ))}
      <button
        className="add-appliance"
        type="button"
        disabled={appliances.length >= 8}
        onClick={() => {
          setAppliances((a) => [
            ...a,
            {
              name: `Appliance ${a.length + 1}`,
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
          ]);
          setDirty(true);
        }}
      >
        <Plus size={18} />
        Add an appliance<span>{appliances.length}/8</span>
      </button>
    </div>
  );
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside
        ref={navRef}
        className={`sidebar ${mobileMenu ? "is-open" : ""}`}
        inert={drawer}
        role={mobileMenu ? "dialog" : undefined}
        aria-modal={mobileMenu ? true : undefined}
        aria-label="Workspace navigation"
        onKeyDown={(e) => {
          if (!mobileMenu) return;
          if (e.key === "Escape") setMobileMenu(false);
          if (e.key === "Tab") {
            const controls = Array.from(
              navRef.current?.querySelectorAll<HTMLElement>("button,a[href]") ||
                [],
            ).filter((el) => el.getClientRects().length > 0);
            const first = controls[0],
              last = controls[controls.length - 1];
            if (e.shiftKey && document.activeElement === first) {
              e.preventDefault();
              last?.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
              e.preventDefault();
              first?.focus();
            }
          }
        }}
      >
        <a
          className="brand"
          href="#overview"
          onClick={() => navigate("overview")}
        >
          <span className="brand-symbol">
            <Zap size={26} fill="currentColor" />
          </span>
          <span>
            current<span className="brand-period">.</span>
            <small>ENERGY IN BALANCE</small>
          </span>
        </a>
        <button
          ref={navCloseRef}
          className="mobile-close icon-button"
          aria-label="Close navigation"
          onClick={() => setMobileMenu(false)}
        >
          <X />
        </button>
        <div className="home-selector">
          <span className="home-avatar">
            <Home size={18} />
          </span>
          <div>
            <strong>My household</strong>
            <small>Energy workspace</small>
          </div>
        </div>
        <span className="nav-caption">YOUR WORKSPACE</span>
        <nav aria-label="Main navigation">
          {views.map((v) => (
            <button
              key={v.id}
              className={view === v.id ? "nav-item active" : "nav-item"}
              aria-current={view === v.id ? "page" : undefined}
              onClick={() => navigate(v.id)}
            >
              <v.icon size={20} />
              {v.label}
              {v.id === "insights" && <span className="nav-tag">WHY</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="energy-tip">
            <div className="tip-icon">
              <Leaf size={21} />
            </div>
            <h3>
              Small shifts.
              <br />
              Smarter energy.
            </h3>
            <p>Give flexible appliances a better time to run.</p>
            <button
              onClick={() => {
                setDrawerTab("tariffs");
                setDrawer(true);
              }}
            >
              Review my tariff <ArrowUpRight size={15} />
            </button>
          </div>
          <button
            className="sidebar-help"
            onClick={() => {
              setDrawerTab("about");
              setDrawer(true);
            }}
          >
            <CircleHelp size={18} />
            How Current works
            <ArrowUpRight size={15} />
          </button>
          <div className="local-indicator">
            <i className="dot green-dot" />
            Local workspace <span>v1.0</span>
          </div>
        </div>
      </aside>
      {mobileMenu && (
        <button
          aria-label="Close navigation"
          className="mobile-scrim"
          onClick={() => setMobileMenu(false)}
        />
      )}
      <div className="workspace" inert={mobileMenu || drawer}>
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="mobile-menu icon-button"
              aria-label="Open navigation"
              onClick={() => setMobileMenu(true)}
            >
              <Menu />
            </button>
            <Home size={15} />
            <ChevronRight size={14} />
            <span>{views.find((v) => v.id === view)?.label}</span>
          </div>
          <div className="topbar-right">
            <span className="history-source">
              <i className="dot blue-dot" />
              {dataMode === "uploaded"
                ? "Uploaded electricity readings"
                : "Recorded electricity history"}
            </span>
            <button
              className="icon-button settings-button"
              aria-label="Open household settings"
              onClick={() => {
                setDrawerTab("household");
                setDrawer(true);
              }}
            >
              <Settings2 size={19} />
            </button>
            <span className="user-avatar" aria-label="My household">
              H
            </span>
          </div>
        </header>
        <main id="main">
          <div className="page-heading">
            <div>
              <span className="eyebrow">
                A LITTLE PLANNING. A LIGHTER BILL.
              </span>
              <h1>
                {view === "overview"
                  ? "Your energy, in balance."
                  : view === "appliances"
                    ? "A plan for every appliance."
                    : view === "insights"
                      ? "Clarity behind every decision."
                      : "Good decisions, kept in view."}
              </h1>
              <p>
                {view === "overview"
                  ? "Understand demand. Find a better time to run. Stay in control."
                  : view === "appliances"
                    ? "Set your priorities, timings and comfort needs."
                    : view === "insights"
                      ? "See the evidence, contributing factors and rules behind your plan."
                      : "Revisit your saved analyses and compare estimated outcomes."}
              </p>
            </div>
            <button
              className="secondary-button"
              onClick={() => {
                setDrawerTab("household");
                setDrawer(true);
              }}
            >
              <SlidersHorizontal size={17} />
              Household settings
            </button>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <Info size={20} />
              <div>
                <strong>Something needs attention</strong>
                <p>{error}</p>
              </div>
              {!boot && (
                <button onClick={() => void load()}>Retry connection</button>
              )}
              <button
                className="icon-button"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={17} />
              </button>
            </div>
          )}
          {boot && (
            <>
              <div
                className="source-switch"
                aria-label="Electricity data source"
              >
                <div>
                  <span className="eyebrow">CHOOSE YOUR ELECTRICITY DATA</span>
                  <p>
                    {dataMode === "uploaded"
                      ? "Prepare an estimate from your latest meter export."
                      : "Explore an example using saved household electricity history."}
                  </p>
                </div>
                <div className="source-buttons">
                  <button
                    disabled={busy || uploading}
                    aria-pressed={dataMode === "recorded"}
                    className={dataMode === "recorded" ? "active" : ""}
                    onClick={() => {
                      setDataMode("recorded");
                      setDirty(true);
                      setError("");
                    }}
                  >
                    <CalendarDays size={16} />
                    Recorded examples
                  </button>
                  <button
                    disabled={busy || uploading}
                    aria-pressed={dataMode === "uploaded"}
                    className={dataMode === "uploaded" ? "active" : ""}
                    onClick={() => {
                      setDataMode("uploaded");
                      setDirty(true);
                      setError("");
                    }}
                  >
                    <Upload size={16} />
                    Use my electricity readings
                  </button>
                </div>
                <button
                  className="secondary-button source-template-button"
                  disabled={busy || uploading}
                  onClick={() => {
                    setDataMode("uploaded");
                    setView("overview");
                    setTemplatesOpen(true);
                    setDirty(true);
                    setError("");
                  }}
                >
                  <Download size={16} />
                  Sample templates
                </button>
              </div>
              {dataMode === "uploaded" && (
                <ReadingsUpload
                  templatesOpen={templatesOpen}
                  onTemplatesOpen={setTemplatesOpen}
                  schema={boot.readings}
                  serviceAvailable={boot.readings.service_available}
                  report={readingsReport}
                  disabled={busy || loading}
                  uploading={uploading}
                  onReport={(report) => {
                    setReadingsReport(report);
                    setDirty(true);
                    setError("");
                  }}
                  onBusyChange={setUploading}
                />
              )}
            </>
          )}
          {boot && (
            <section className="context-bar" aria-label="Analysis settings">
              <div className="context-icon">
                <CalendarDays size={20} />
              </div>
              {dataMode === "uploaded" ? (
                <div className="history-picker upload-context">
                  <span>Estimate from my readings</span>
                  <strong>
                    {readingsReport?.origin
                      ? `${date(readingsReport.origin)} · ${clock(readingsReport.origin)} · ${readingsReport.timezone}`
                      : "Upload and check a CSV to begin"}
                  </strong>
                  <small>
                    {readingsReport?.sample_ready ? "Sample forecast available · original sample dates" : readingsReport?.can_analyze
                      ? "Ready to forecast"
                      : readingsReport
                        ? "No forecast generated · review the checks above"
                        : "No readings selected"}
                  </small>
                </div>
              ) : (
                <label className="history-picker">
                  <span>Electricity history</span>
                  <select
                    aria-label="Select recorded electricity history"
                    disabled={busy}
                    value={caseId}
                    onChange={(e) => {
                      setCaseId(e.target.value);
                      setDirty(true);
                    }}
                  >
                    {boot.cases.map((c) => (
                      <option key={c.id} value={c.id}>
                        {date(c.origin)} · {clock(c.origin)}
                        {c.explanation ? " · with explanation" : ""}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <div className="context-separator" />
              <div className="context-fact">
                <Users size={17} />
                <span>
                  <b>{scenario?.occupancy}</b> people home
                </span>
              </div>
              <div className="context-fact">
                <Thermometer size={17} />
                <span>
                  <b>{scenario?.temperature_c}°</b> outside
                </span>
              </div>
              <button
                className="primary-button"
                disabled={!canRun}
                onClick={() => void analyze()}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  <Sparkles size={17} />
                )}{" "}
                {busy
                  ? "Finding your plan…"
                  : dataMode === "uploaded" && readingsReport?.sample_ready
                    ? "Generate sample forecast"
                  : dataMode === "uploaded" && !readingsReport?.can_analyze
                    ? "Estimate unavailable"
                    : run
                      ? "Update analysis"
                      : "Analyze my energy"}
                {!busy && <ArrowRight size={17} />}
              </button>
            </section>
          )}
          {dirty && resultMatchesSource && run && (
            <div className="stale-banner" role="status">
              <RefreshCw size={16} />
              <span>
                Settings changed. Results below still belong to your last
                analysis.
              </span>
              <button disabled={!canRun} onClick={() => void analyze()}>
                Update results <ArrowRight size={14} />
              </button>
            </div>
          )}
          {loading && (
            <div className="loading-state" role="status">
              <LoaderCircle className="spin" />
              Opening your energy workspace…
            </div>
          )}
          {!loading && view === "overview" && (
            <>
              {resultMatchesSource && run && f && p ? (
                <section
                  ref={resultRef}
                  tabIndex={-1}
                  className="results"
                  aria-label="Latest analysis results"
                >
                  {run.source_mode === "uploaded_readings" && (
                    <div className="uploaded-result-meta">
                      <CheckCircle2 size={17} />
                      <span>
                        {f.is_sample ? "SAMPLE FORECAST · Synthetic data. This is not your actual current electricity use. File: " : "Saved estimate from "}
                        <strong>{run.uploaded_readings?.file_name}</strong> ·
                        readings through {date(f.origin)} {clock(f.origin)} (
                        {run.uploaded_readings?.timezone}). Generated{" "}
                        {new Date(run.created_at).toLocaleString("en-IN")}.
                        {f.input_scope && <span> {f.input_scope}</span>}
                      </span>
                    </div>
                  )}
                  <div className="overview-grid">
                    <section className="forecast-hero">
                      <div className="hero-top">
                        <span>
                          <i className="dot" />
                          {f.is_sample ? "SAMPLE DEMAND ESTIMATE" : "YOUR NEXT DEMAND ESTIMATE"}
                        </span>
                        <Pill tone="hero">
                          {high ? "High demand" : "Normal demand"}
                          <Activity size={13} />
                        </Pill>
                      </div>
                      <div className="hero-value">
                        {f.forecast_kw.toFixed(3)}
                        <span>kW</span>
                      </div>
                      <p>
                        Average power · {date(f.target_end)} ·{" "}
                        {clock(f.target_interval_start)}–{clock(f.target_end)}
                      </p>
                      <div className="hero-scale">
                        <span
                          style={{
                            width: `${Math.min(100, (f.forecast_kw / f.threshold_kw) * 72)}%`,
                          }}
                        />
                        <i style={{ left: "72%" }} />
                      </div>
                      <div className="hero-scale-label">
                        <span>Lower demand</span>
                        <span>High from {f.threshold_kw.toFixed(2)} kW</span>
                      </div>
                      <div className="hero-bottom">
                        <ShieldCheck size={17} />
                        <div>
                          <strong>
                            Confidence: {run.confidence.label.toLowerCase()}
                          </strong>
                          <small>
                            No calibrated range is provided for this source.
                          </small>
                        </div>
                        <button
                          aria-label="Read about forecast confidence"
                          onClick={() => {
                            setDrawerTab("about");
                            setDrawer(true);
                          }}
                        >
                          <Info size={18} />
                        </button>
                      </div>
                      <Zap className="hero-watermark" />
                    </section>
                    <section className="metric-card savings-card">
                      <div className="metric-top">
                        <span className="metric-icon orange-icon">
                          <ArrowDownRight size={21} />
                        </span>
                        <span className="eyebrow">ESTIMATED JOB SAVINGS</span>
                      </div>
                      <div className="metric-value">
                        {money(p.estimated_saving)}
                      </div>
                      <p>
                        {p.estimated_saving == null
                          ? "Review appliance timings to estimate savings"
                          : p.estimated_saving === 0
                            ? "No lower-cost timing found"
                            : "By choosing a better time to run"}
                      </p>
                      <div className="cost-comparison">
                        <div>
                          <span>Earliest feasible plan</span>
                          <b>{money(p.baseline_cost)}</b>
                        </div>
                        <div>
                          <span>Recommended plan</span>
                          <b>{money(p.estimated_cost)}</b>
                        </div>
                      </div>
                      <small>
                        Scenario estimate for scheduled jobs.
                        <br />
                        Actual bill savings are not measured.
                      </small>
                    </section>
                    <section className="metric-card plan-card">
                      <div className="metric-top">
                        <span className="metric-icon blue-icon">
                          <CalendarDays size={21} />
                        </span>
                        <span className="eyebrow">YOUR APPLIANCE PLAN</span>
                      </div>
                      <div
                        className={`plan-status ${p.status !== "optimal" ? "needs-attention" : "normal"}`}
                      >
                        {p.status === "optimal" ? (
                          <CheckCircle2 size={22} />
                        ) : (
                          <Info size={22} />
                        )}
                        <strong>{statusLabel(p.status)}</strong>
                      </div>
                      <div className="plan-stat">
                        <b>{p.schedule.length.toString().padStart(2, "0")}</b>
                        <span>
                          jobs scheduled around
                          <br />
                          your household priorities
                        </span>
                      </div>
                      <div className="plan-mini-rule">
                        <ShieldCheck size={16} />
                        {
                          run.recommendations.filter(
                            (r) => r.Kind === "protected",
                          ).length
                        }{" "}
                        appliances protected
                      </div>
                      <button
                        className="text-button"
                        onClick={() => navigate("insights")}
                      >
                        See why this works <ArrowRight size={15} />
                      </button>
                    </section>
                  </div>
                  <div className="analysis-columns">
                    <div>
                      <DemandChart run={run} />
                      <ScheduleView
                        run={run}
                        onAdjust={() => navigate("appliances")}
                      />
                    </div>
                    <div>
                      <section className="panel next-steps">
                        <PanelTitle
                          eyebrow="YOUR NEXT STEPS"
                          title="A few thoughtful changes"
                        />
                        <p className="subtext">
                          Recommendations from your latest analysis.
                        </p>
                        <div className="recommendation-list">
                          {run.recommendations.map((r) => (
                            <div key={r.Appliance} className="recommendation">
                              <span
                                className={`appliance-icon ${r.Kind === "protected" ? "mint" : ""}`}
                              >
                                <ApplianceIcon name={r.Appliance} />
                              </span>
                              <div>
                                <strong>{r.Appliance}</strong>
                                <span
                                  className={
                                    r.Action === "Delay" ? "warm-text" : ""
                                  }
                                >
                                  {r.Action}
                                  {r.Suggested_start
                                    ? ` · ${clock(r.Suggested_start)}`
                                    : ""}
                                </span>
                                <p>{r.Reason}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                        <button
                          className="full-link"
                          onClick={() => navigate("insights")}
                        >
                          Explore the reasoning <ArrowRight size={16} />
                        </button>
                      </section>
                      <section className="tariff-card">
                        <div className="tariff-title">
                          <span className="metric-icon">
                            <Zap size={18} />
                          </span>
                          <h3>Timing makes a difference.</h3>
                        </div>
                        <p>Your scenario’s electricity prices</p>
                        <div className="tariff-rates">
                          <div>
                            <small>OFF-PEAK</small>
                            <b>
                              {money(run.scenario.offpeak_rate)}
                              <span>/ kWh</span>
                            </b>
                          </div>
                          <div>
                            <small>PEAK HOURS</small>
                            <b>
                              {money(run.scenario.peak_rate)}
                              <span>/ kWh</span>
                            </b>
                          </div>
                        </div>
                        <div className="tariff-hours">
                          <Clock3 size={15} />
                          {run.scenario.peak_start_hour ===
                          run.scenario.peak_end_hour
                            ? "No peak window configured"
                            : `Peak: ${String(run.scenario.peak_start_hour).padStart(2, "0")}:00–${String(run.scenario.peak_end_hour).padStart(2, "0")}:00`}
                        </div>
                        <button
                          onClick={() => {
                            setDrawerTab("tariffs");
                            setDrawer(true);
                          }}
                        >
                          Adjust tariff <ArrowUpRight size={15} />
                        </button>
                      </section>
                    </div>
                  </div>
                  <div className="result-footer">
                    <span>
                      <CheckCircle2 size={14} />
                      Saved {new Date(run.created_at).toLocaleString("en-IN")} ·
                      History from {date(f.origin)}
                    </span>
                    <div>
                      <a href={`/api/history/${run.id}/export/json`} download>
                        <Download size={14} />
                        Analysis
                      </a>
                      <a
                        href={`/api/history/${run.id}/export/schedule`}
                        download
                      >
                        <Download size={14} />
                        Schedule CSV
                      </a>
                    </div>
                  </div>
                </section>
              ) : dataMode === "uploaded" ? (
                <section className="panel empty upload-empty">
                  <Upload />
                  <h2>Your current-date estimate will appear here</h2>
                  <p>
                    {readingsReport?.can_analyze
                      ? "Your file is ready. Use “Analyze my energy” to generate a forecast and appliance plan."
                      : readingsReport
                        ? "Your upload checks are shown above. A forecast and savings will appear only when the readings and forecasting connection are ready."
                        : "Upload recent meter readings using the form above. The app will check their date range, completeness and freshness before generating an estimate."}
                  </p>
                  <small>
                    Recorded examples remain available in Analysis history.
                  </small>
                </section>
              ) : (
                <section className="welcome-panel">
                  <div className="welcome-art">
                    <div className="orbit one" />
                    <div className="orbit two" />
                    <Home size={66} />
                    <span>
                      <Zap size={24} />
                    </span>
                  </div>
                  <div>
                    <span className="eyebrow">MEET YOUR ENERGY WORKSPACE</span>
                    <h2>A better rhythm for your home.</h2>
                    <p>
                      Choose an electricity history, add your household
                      preferences, and find a practical appliance plan with
                      clear reasons behind it.
                    </p>
                    <div className="welcome-features">
                      <span>
                        <Gauge size={18} />
                        Demand estimate
                      </span>
                      <span>
                        <ShieldCheck size={18} />
                        Household protection
                      </span>
                      <span>
                        <Lightbulb size={18} />
                        Clear explanations
                      </span>
                    </div>
                    <button
                      className="primary-button"
                      disabled={!canRun}
                      onClick={() => void analyze()}
                    >
                      {busy ? (
                        <LoaderCircle className="spin" size={17} />
                      ) : (
                        <Sparkles size={17} />
                      )}
                      Create my first analysis <ArrowRight size={17} />
                    </button>
                    <small>
                      Uses genuine recorded forecasts. No live meter connection
                      is configured.
                    </small>
                  </div>
                </section>
              )}
            </>
          )}
          {!loading && view === "appliances" && (
            <>
              <div className="appliances-intro">
                <ShieldCheck size={24} />
                <div>
                  <h3>Your priorities come first.</h3>
                  <p>
                    Essential appliances stay protected. Already-running and
                    high-priority appliances are not delayed automatically.
                  </p>
                </div>
                <Pill tone="neutral">{appliances.length} appliances</Pill>
              </div>
              <form
                className="panel appliance-page"
                onSubmit={(e) => {
                  e.preventDefault();
                  void analyze();
                }}
              >
                <PanelTitle
                  title="Household appliances"
                  action={
                    <button
                      className="primary-button"
                      disabled={!canRun}
                      type="submit"
                    >
                      {busy ? (
                        <LoaderCircle className="spin" size={16} />
                      ) : (
                        <Check size={16} />
                      )}
                      Analyze changes
                    </button>
                  }
                />
                <fieldset className="editor-fieldset" disabled={busy}>
                  {applianceEditor}
                </fieldset>
              </form>
            </>
          )}
          {!loading &&
            view === "insights" &&
            (resultMatchesSource && run ? (
              <div className="insights-grid">
                <Explanations run={run} />
                <section className="panel confidence-panel">
                  <ShieldCheck size={25} />
                  <div>
                    <h2>Understand the limits, too.</h2>
                    <p>
                      {run.confidence.reason} Household preferences influence
                      the plan and estimated costs. Changing them does not
                      change the electricity readings used for the forecast.
                    </p>
                    <p>
                      Only the forecast interval is avoided when demand is high.
                      Later start times use your tariff and appliance
                      constraints.
                    </p>
                  </div>
                </section>
              </div>
            ) : (
              <div className="panel empty">
                <Lightbulb />
                <h2>Start with an analysis</h2>
                <p>
                  Your forecast evidence and household rule checks will appear
                  here.
                </p>
                <button
                  className="primary-button"
                  onClick={() => navigate("overview")}
                >
                  Go to overview <ArrowRight size={16} />
                </button>
              </div>
            ))}
          {!loading && view === "history" && (
            <section className="panel history-panel">
              <PanelTitle
                title="Saved analyses"
                action={
                  <Pill tone="neutral">{saved.length} saved locally</Pill>
                }
              />
              <div className="history-toolbar">
                <input
                  type="search"
                  placeholder="Search by history date or run ID"
                  aria-label="Search saved analyses"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <select
                  aria-label="Filter analysis status"
                  value={historyFilter}
                  onChange={(e) => setHistoryFilter(e.target.value)}
                >
                  <option value="all">All results</option>
                  <option value="optimal">Plan ready</option>
                  <option value="attention">Needs attention</option>
                </select>
              </div>
              {saved.length ? (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Saved analysis</th>
                        <th>Electricity history</th>
                        <th>Forecast</th>
                        <th>Estimated saving</th>
                        <th>Status</th>
                        <th>
                          <span className="sr-only">Open</span>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {saved
                        .filter(
                          (r) =>
                            (r.forecast.origin + " " + r.id)
                              .toLowerCase()
                              .includes(search.toLowerCase()) &&
                            (historyFilter === "all" ||
                              (historyFilter === "optimal"
                                ? r.plan.status === "optimal"
                                : r.plan.status !== "optimal")),
                        )
                        .map((r) => (
                          <tr key={r.id}>
                            <td>
                              <strong>
                                {new Date(r.created_at).toLocaleDateString(
                                  "en-GB",
                                  { day: "numeric", month: "short" },
                                )}{" "}
                                ·{" "}
                                {new Date(r.created_at).toLocaleTimeString(
                                  "en-GB",
                                  { hour: "2-digit", minute: "2-digit" },
                                )}
                              </strong>
                              <small>#{r.id.slice(0, 8)}</small>
                            </td>
                            <td>
                              {date(r.forecast.origin)}
                              <small>
                                {clock(r.forecast.origin)} ·{" "}
                                {r.source_mode === "uploaded_readings"
                                  ? r.forecast.is_sample ? "synthetic sample" : "uploaded readings"
                                  : "recorded history"}
                              </small>
                            </td>
                            <td>{num(r.forecast.forecast_kw)} kW</td>
                            <td className="blue-text">
                              {money(r.plan.estimated_saving)}
                            </td>
                            <td>
                              <Pill
                                tone={
                                  r.plan.status === "optimal"
                                    ? "green"
                                    : "orange"
                                }
                              >
                                {statusLabel(r.plan.status)}
                              </Pill>
                            </td>
                            <td>
                              <button
                                className="icon-button"
                                aria-label={`Open analysis ${r.id.slice(0, 8)}`}
                                onClick={() => void restore(r.id)}
                              >
                                <ArrowUpRight size={19} />
                              </button>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                  {!saved.some(
                    (r) =>
                      (r.forecast.origin + " " + r.id)
                        .toLowerCase()
                        .includes(search.toLowerCase()) &&
                      (historyFilter === "all" ||
                        (historyFilter === "optimal"
                          ? r.plan.status === "optimal"
                          : r.plan.status !== "optimal")),
                  ) && (
                    <div className="empty compact">
                      <p>No analyses match your filters.</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="empty">
                  <FileClock />
                  <h3>Your history starts here.</h3>
                  <p>
                    Completed analyses are saved on this computer so you can
                    revisit and export them.
                  </p>
                  <button
                    className="text-button"
                    onClick={() => navigate("overview")}
                  >
                    Create an analysis <ArrowRight size={16} />
                  </button>
                </div>
              )}
              <p className="microcopy">
                Estimates from different history dates or household scenarios
                are not measured bill savings. The latest 100 analyses are
                listed.
              </p>
            </section>
          )}
          <footer className="page-footer">
            <span className="footer-brand">
              <Zap size={14} />
              current.
            </span>
            <p>Recommendations for your home. Decisions stay with you.</p>
            <button
              onClick={() => {
                setDrawerTab("about");
                setDrawer(true);
              }}
            >
              Data & limitations <ExternalLink size={12} />
            </button>
          </footer>
        </main>
      </div>
      {drawer && (
        <div
          className="drawer-overlay"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setDrawer(false);
          }}
        >
          <div
            ref={dialogRef}
            className="settings-drawer"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
            onKeyDown={(e) => {
              if (e.key === "Escape" && !busy) setDrawer(false);
              if (e.key === "Tab") {
                const all = dialogRef.current?.querySelectorAll<HTMLElement>(
                  'button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),summary,[tabindex="0"]',
                );
                if (!all?.length) return;
                const first = all[0],
                  last = all[all.length - 1];
                if (e.shiftKey && document.activeElement === first) {
                  e.preventDefault();
                  last.focus();
                } else if (!e.shiftKey && document.activeElement === last) {
                  e.preventDefault();
                  first.focus();
                }
              }
            }}
          >
            <div className="drawer-header">
              <div>
                <span className="eyebrow">MAKE IT YOURS</span>
                <h2 id="settings-title">Household settings</h2>
              </div>
              <button
                ref={closeRef}
                className="icon-button"
                aria-label="Close settings"
                disabled={busy}
                onClick={() => setDrawer(false)}
              >
                <X />
              </button>
            </div>
            <div className="drawer-tabs">
              {["household", "tariffs", "appliances", "about"].map((t) => (
                <button
                  key={t}
                  aria-pressed={drawerTab === t}
                  className={drawerTab === t ? "active" : ""}
                  onClick={() => setDrawerTab(t)}
                >
                  {t === "about"
                    ? "About the data"
                    : t[0].toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
            <form
              className="drawer-form"
              onSubmit={(e) => {
                e.preventDefault();
                void analyze();
              }}
            >
              <fieldset disabled={busy} className="drawer-content">
                {error && (
                  <div className="error-banner" role="alert">
                    <Info size={18} />
                    <p>{error}</p>
                  </div>
                )}
                {drawerTab === "household" && (
                  <>
                    <div className="form-intro">
                      <Home size={26} />
                      <h3>A little context for a better plan.</h3>
                      <p>
                        These preferences guide the appliance rules and
                        scheduling. The forecast follows your selected
                        electricity history.
                      </p>
                    </div>
                    <div className="form-grid">
                      <ScenarioField
                        scenario={scenario}
                        onChange={updateScenario}
                        label="People at home"
                        field="occupancy"
                        min={0}
                        max={20}
                      />
                      <ScenarioField
                        scenario={scenario}
                        onChange={updateScenario}
                        label="Outside temperature"
                        field="temperature_c"
                        min={-50}
                        max={60}
                        step={0.5}
                        suffix="°C"
                      />
                      <ScenarioField
                        scenario={scenario}
                        onChange={updateScenario}
                        label="Planning horizon"
                        field="horizon_hours"
                        min={1}
                        max={12}
                        suffix="hours"
                      />
                      <ScenarioField
                        scenario={scenario}
                        onChange={updateScenario}
                        label="Parallel jobs"
                        field="max_parallel_jobs"
                        min={1}
                        max={6}
                      />
                    </div>
                    <div className="evidence-note">
                      <ShieldCheck size={20} />
                      <p>
                        Parallel jobs counts scheduled jobs only. Essential
                        appliances and comfort rules are evaluated separately.
                      </p>
                    </div>
                  </>
                )}
                {drawerTab === "tariffs" && (
                  <>
                    <div className="form-intro">
                      <Zap size={26} />
                      <h3>Give every hour the right price.</h3>
                      <p>
                        Enter rates in ₹ per kWh. These are scenario settings,
                        not a connected utility tariff.
                      </p>
                    </div>
                    <div className="form-grid">
                      <ScenarioField
                        scenario={scenario}
                        onChange={updateScenario}
                        label="Off-peak price"
                        field="offpeak_rate"
                        min={0}
                        max={10000}
                        step={0.01}
                        suffix="₹ / kWh"
                      />
                      <ScenarioField
                        scenario={scenario}
                        onChange={updateScenario}
                        label="Peak price"
                        field="peak_rate"
                        min={0}
                        max={10000}
                        step={0.01}
                        suffix="₹ / kWh"
                      />
                      <ScenarioField
                        scenario={scenario}
                        onChange={updateScenario}
                        label="Peak starts at"
                        field="peak_start_hour"
                        min={0}
                        max={23}
                        suffix="hour (0–23)"
                      />
                      <ScenarioField
                        scenario={scenario}
                        onChange={updateScenario}
                        label="Peak ends at"
                        field="peak_end_hour"
                        min={0}
                        max={23}
                        suffix="hour (0–23)"
                      />
                    </div>
                    <div className="evidence-note">
                      <Clock3 size={20} />
                      <p>
                        Overnight peak windows are supported. Equal start and
                        end hours mean no peak period. Costs exclude protected
                        appliances, fixed utility charges and unmodelled
                        household use.
                      </p>
                    </div>
                  </>
                )}
                {drawerTab === "appliances" && applianceEditor}
                {drawerTab === "about" && (
                  <div className="about-content">
                    <span className="about-symbol">
                      <Zap size={34} />
                    </span>
                    <h3>Clear estimates. Honest evidence.</h3>
                    <p>
                      Current connects your electricity history with household
                      rules and a constrained scheduling search.
                    </p>
                    <dl>
                      <dt>Electricity source</dt>
                      <dd>
                        {dataMode === "uploaded"
                          ? "User-supplied minute readings. File checks run locally; new forecasts require a compatible forecasting connection."
                          : boot?.mode === "recorded_forecasts"
                            ? "Saved forecasts from recorded household electricity data (2006–2010)."
                            : "Connected saved-history forecasting bundle."}{" "}
                        No live meter is connected.
                      </dd>
                      <dt>Confidence</dt>
                      <dd>
                        No calibrated score or interval is supplied. A
                        percentage would imply evidence this source does not
                        provide.
                      </dd>
                      <dt>Explanations</dt>
                      <dd>
                        Verified contributions appear only for matching
                        histories. Rule traces show the rules actually evaluated
                        for your scenario.
                      </dd>
                      <dt>Savings</dt>
                      <dd>
                        Estimated scheduled-job cost difference versus a
                        feasible earliest-start plan using the same constraints.
                        This is not a measured reduction in your bill.
                      </dd>
                      <dt>Appliance control</dt>
                      <dd>
                        Current recommends start times. It does not switch
                        appliances.
                      </dd>
                      <dt>Local history</dt>
                      <dd>
                        Analyses are saved on this computer. This local
                        workspace has no account service or cloud
                        synchronization.
                      </dd>
                    </dl>
                    {boot?.integration_gap && (
                      <div className="evidence-note">
                        <Info size={19} />
                        <p>{boot.integration_gap}</p>
                      </div>
                    )}
                  </div>
                )}
              </fieldset>
              <div className="drawer-footer">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => setDrawer(false)}
                  disabled={busy}
                >
                  Keep editing later
                </button>
                <button
                  className="primary-button"
                  type="submit"
                  disabled={!canRun}
                >
                  {busy ? (
                    <LoaderCircle className="spin" size={17} />
                  ) : (
                    <Sparkles size={17} />
                  )}
                  Analyze my energy
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
