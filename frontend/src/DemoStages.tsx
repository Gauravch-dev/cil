import { useState } from "react";
import {
  ArrowRight,
  BrainCircuit,
  CalendarDays,
  Check,
  CheckCircle2,
  Database,
  GitBranch,
  Lightbulb,
  LoaderCircle,
  Network,
  Play,
  Settings2,
  ShieldCheck,
  Zap,
} from "lucide-react";
import type { Bootstrap, Run } from "./types";
import { DemandChart, ScheduleView } from "./ResultComponents";
import PlanResult from "./PlanResult";
import {
  day,
  downloadEvidence,
  Evidence,
  friendly,
  money,
  Notice,
  Stat,
  time,
} from "./workspace";

export const stages = [
  { title: "Set the scene", sub: "Choose inputs", icon: Settings2 },
  { title: "Prepare the data", sub: "History & features", icon: Database },
  { title: "Make a forecast", sub: "DL & ML roles", icon: BrainCircuit },
  { title: "Understand the home", sub: "Ontology", icon: Network },
  { title: "Apply the rules", sub: "Reasoning", icon: GitBranch },
  { title: "Find a schedule", sub: "A* search", icon: CalendarDays },
  { title: "Explain the forecast", sub: "SHAP", icon: Lightbulb },
  { title: "See the outcome", sub: "Results & evidence", icon: CheckCircle2 },
];
function ShapView({ evidence }: { evidence: Evidence }) {
  const [all, setAll] = useState(false);
  const e = evidence.shap;
  if (!e.available) return <Notice tone="orange">{e.reason}</Notice>;
  const factors = [...(e.factors || [])].sort(
    (a, b) => Math.abs(b.value_kw) - Math.abs(a.value_kw),
  );
  const shown = all
    ? factors
    : [
        ...factors.slice(0, 8),
        ...(factors.length > 8
          ? [
              {
                name: "Other input groups",
                value_kw: factors.slice(8).reduce((s, x) => s + x.value_kw, 0),
              },
            ]
          : []),
      ];
  const scale = Math.max(0.01, ...shown.map((x) => Math.abs(x.value_kw)));
  return (
    <>
      <div className="s-equation">
        <div>
          <small>Reference output</small>
          <b>{e.reference_kw?.toFixed(3)} kW</b>
        </div>
        <span>+</span>
        <div>
          <small>SHAP contributions</small>
          <b>
            {(e.factors || []).reduce((s, x) => s + x.value_kw, 0).toFixed(3)}{" "}
            kW
          </b>
        </div>
        <span>≈</span>
        <div className="s-equation-result">
          <small>
            {e.raw_forecast_kw != null && e.raw_forecast_kw < 0
              ? "Raw output before clipping"
              : "This prediction"}
          </small>
          <b>{(e.raw_forecast_kw ?? evidence.forecast_kw).toFixed(3)} kW</b>
        </div>
      </div>
      <div className="s-inline-between">
        <p className="s-help">
          Orange raises the forecast; blue lowers it relative to the reference.
        </p>
        <button className="s-text" onClick={() => setAll(!all)}>
          {all ? "Show strongest" : "Show all groups"}
        </button>
      </div>
      <div className="s-shap-bars">
        {shown.map((f) => (
          <details key={f.name}>
            <summary>
              <span>{friendly(f.name)}</span>
              <span className="s-shap-track">
                <i
                  className={f.value_kw >= 0 ? "positive" : "negative"}
                  style={{ width: `${(Math.abs(f.value_kw) / scale) * 100}%` }}
                />
              </span>
              <b className={f.value_kw >= 0 ? "s-orange" : "s-blue"}>
                {f.value_kw >= 0 ? "+" : ""}
                {f.value_kw.toFixed(3)}
              </b>
            </summary>
            <p>
              {f.value_kw >= 0 ? "Raises" : "Lowers"} the model output by
              approximately {Math.abs(f.value_kw).toFixed(3)} kW relative to the
              chosen training reference. This is an association, not proof of
              physical causation.
            </p>
          </details>
        ))}
      </div>
      <Notice tone={e.reconstruction_verified ? "green" : "orange"}>
        <strong>
          {e.reconstruction_verified
            ? "Reconstruction verified"
            : "Approximation outside tolerance"}
          .
        </strong>{" "}
        Reference + all contributions reconstruct the prediction with residual{" "}
        {e.residual_kw?.toExponential(2)} kW.
      </Notice>
      <button
        className="s-button secondary"
        onClick={() => downloadEvidence(evidence)}
      >
        Download model & SHAP evidence
      </button>
      <details className="s-technical-details">
        <summary>Method, reference and limitations</summary>
        <p>{e.note}</p>
        <p>
          {e.group_count}{" "}
          {e.method === "TreeSHAP"
            ? "learned/context features"
            : "input groups"}
          {e.permutations != null &&
            ` · ${e.permutations} sampled permutations`}
          {e.reference_training_index != null &&
            ` · training target index ${e.reference_training_index}`}
          . This is computed for the displayed model output. Saved Integrated
          Gradients are not relabelled as SHAP.
        </p>
      </details>
    </>
  );
}
export default function DemoStages({
  stage,
  run,
  boot,
  evidence,
  evidenceBusy,
  evidenceError,
  onEvidence,
  onAdjust,
}: {
  stage: number;
  run: Run;
  boot: Bootstrap;
  evidence: Evidence | null;
  evidenceBusy: boolean;
  evidenceError: string;
  onEvidence: () => void;
  onAdjust: () => void;
}) {
  const [selected, setSelected] = useState(0),
    [tab, setTab] = useState("dl");
  const r =
    run.recommendations[Math.min(selected, run.recommendations.length - 1)];
  const picker = (
    <label className="s-field s-appliance-picker">
      <span>Inspect an appliance</span>
      <select
        value={Math.min(selected, run.recommendations.length - 1)}
        onChange={(e) => setSelected(Number(e.target.value))}
      >
        {run.recommendations.map((x, i) => (
          <option key={x.Appliance} value={i}>
            {x.Appliance}
          </option>
        ))}
      </select>
    </label>
  );
  const pending = !evidence ? (
    <div className="s-empty" role="status">
      <BrainCircuit size={35} />
      <h3>
        {evidenceBusy
          ? "Computing model evidence…"
          : "Inspect the actual model"}
      </h3>
      <p>
        {evidenceError ||
          "Re-run the frozen model, inspect activations and compute SHAP for this exact forecast. No training is performed."}
      </p>
      <button className="s-button" disabled={evidenceBusy} onClick={onEvidence}>
        {evidenceBusy ? (
          <LoaderCircle className="spin" size={16} />
        ) : (
          <Play size={16} />
        )}{" "}
        {evidenceBusy ? "Computing…" : "Compute model & SHAP evidence"}
      </button>
    </div>
  ) : null;
  if (stage === 1)
    return (
      <>
        <div className="s-three-stats">
          <Stat
            label="Past readings"
            value="96 intervals"
            note="24 hours at 15-minute resolution"
          />
          <Stat
            label="History channels"
            value="6"
            note="Power, observation flag and calendar"
          />
          <Stat
            label="Context features"
            value="17"
            note="Recent, daily and weekly patterns"
          />
        </div>
        <DemandChart run={run} />
        <details className="s-technical-details">
          <summary>Inspect preprocessing and features</summary>
          <div className="s-process-row">
            {[
              "Minute readings",
              "Completed means",
              "Past-only gap filling",
              "Frozen scaling",
              "History + context",
            ].map((x, i) => (
              <span key={x}>
                {i > 0 && <ArrowRight size={13} />}
                <b>{x}</b>
              </span>
            ))}
          </div>
          <p>
            Only observations available at the forecast origin are used. Short
            gaps use at most four previous bins; future meter values are never
            invented. Scalers are transformed, never fitted.
          </p>
          <p>
            Source:{" "}
            {run.forecast.is_sample
              ? "marked synthetic sample"
              : run.source_mode === "uploaded_readings"
                ? "uploaded meter export"
                : "recorded household dataset"}
            . History ends {run.forecast.origin}; target ends{" "}
            {run.forecast.target_end}.
          </p>
          {evidence && (
            <>
              <div className="s-code-tags">
                {evidence.history_channels.map((x) => (
                  <code key={x}>{x}</code>
                ))}
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Context feature</th>
                      <th>Scaled value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evidence.context_features.map((x) => (
                      <tr key={x.name}>
                        <td>{friendly(x.name)}</td>
                        <td>{x.scaled_value.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </details>
      </>
    );
  if (stage === 2)
    return (
      <>
        <div
          className="s-tabs"
          role="tablist"
          aria-label="Forecasting technology"
        >
          <button
            role="tab"
            aria-selected={tab === "dl"}
            onClick={() => setTab("dl")}
          >
            Deep learning
          </button>
          <button
            role="tab"
            aria-selected={tab === "ml"}
            onClick={() => setTab("ml")}
          >
            ML & baselines
          </button>
        </div>
        {tab === "dl" ? (
          <>
            <div className="s-model-result">
              <span className="s-small-icon">
                <BrainCircuit size={24} />
              </span>
              <div>
                <small>
                  {evidence?.classical_ml.available
                    ? evidence.classical_ml.model
                    : run.forecast.model}
                </small>
                <strong>
                  {run.forecast.forecast_kw.toFixed(3)} <span>kW</span>
                </strong>
                <p>
                  {day(run.forecast.target_end)} ·{" "}
                  {time(run.forecast.target_interval_start)}–
                  {time(run.forecast.target_end)}
                </p>
              </div>
              <span className="s-tag">
                {evidence?.prediction_reexecuted
                  ? "Re-executed & matched"
                  : "Saved analysis output"}
              </span>
            </div>
            <div className="s-layer-flow">
              {(evidence?.classical_ml.available
                ? [
                    "96 × 6 history",
                    "LSTM · 32 units",
                    "LSTM · 16 patterns",
                    "+ 17 context",
                    evidence.classical_ml.model?.replace("LSTM + ", "") ||
                      "Classical regressor",
                    "Forecast · 1",
                  ]
                : [
                    "96 × 6 history",
                    "LSTM · 32 units",
                    "LSTM · 16 units",
                    "+ 17 context",
                    "Dense · 32",
                    "Forecast · 1",
                  ]
              ).map((x, i) => (
                <div key={x}>
                  <span>{String(i + 1).padStart(2, "0")}</span>
                  <strong>{x}</strong>
                  {i < 5 && <ArrowRight size={17} />}
                </div>
              ))}
            </div>
            <p className="s-help">
              {evidence?.classical_ml.available
                ? "The frozen LSTM extracts 16 learned patterns. The trained classical regressor combines them with 17 context features to produce this forecast."
                : "The LSTM learns temporal patterns; context and a dense output turn those patterns into one estimate."}{" "}
              Household preferences enter the later policy stages.
            </p>
            {pending}
            {evidence && (
              <details className="s-technical-details">
                <summary>Inspect LSTM activations</summary>
                <p>
                  Actual internal responses: 32 units across 96 time steps.
                  These are not attention weights, feature contributions or
                  confidence.
                </p>
                <svg
                  className="s-activation"
                  viewBox="0 0 768 256"
                  role="img"
                  aria-label="LSTM activation heatmap, 32 units and 96 historical intervals"
                >
                  {evidence.activation.map((row, i) =>
                    row.map((v, j) => (
                      <rect
                        key={`${i}-${j}`}
                        x={j * 8}
                        y={i * 8}
                        width={7.5}
                        height={7.5}
                        fill={
                          v >= 0
                            ? `rgba(237,125,50,${0.12 + Math.min(1, Math.abs(v)) * 0.88})`
                            : `rgba(41,91,205,${0.12 + Math.min(1, Math.abs(v)) * 0.88})`
                        }
                      >
                        <title>
                          Unit {i + 1}, step {j + 1}: {v.toFixed(4)}
                        </title>
                      </rect>
                    )),
                  )}
                </svg>
                <p className="s-help">
                  Oldest → newest. Model fingerprint{" "}
                  <code>{evidence.model_sha256.slice(0, 16)}…</code>
                </p>
              </details>
            )}
          </>
        ) : (
          <>
            <Notice tone={evidence?.classical_ml.available ? "green" : "blue"}>
              <strong>
                {evidence?.classical_ml.available
                  ? `${evidence.classical_ml.model} executed.`
                  : "Inspect the prediction stage."}
              </strong>{" "}
              {evidence?.classical_ml.reason ||
                "Compute evidence to inspect the model used for this saved analysis."}
            </Notice>
            {evidence?.classical_ml.available && (
              <>
                <div className="s-three-stats">
                  <Stat
                    label="Learned LSTM patterns"
                    value={evidence.classical_ml.latent_features}
                  />
                  <Stat
                    label="Context features"
                    value={evidence.classical_ml.context_features}
                  />
                  <Stat
                    label="Regressor inputs"
                    value={evidence.classical_ml.feature_count}
                  />
                </div>
                <h3>Selected using validation data</h3>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>ML candidate</th>
                        <th>Validation MAE</th>
                        <th>Validation RMSE</th>
                        <th>Selection</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evidence.classical_ml.validation?.map((x) => (
                        <tr key={x.Model}>
                          <td>{x.Model}</td>
                          <td>{x.MAE_kW.toFixed(3)} kW</td>
                          <td>{x.RMSE_kW.toFixed(3)} kW</td>
                          <td>
                            {x.Model === evidence.classical_ml.model
                              ? "Selected"
                              : "Compared"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <details className="s-technical-details">
                  <summary>Inspect the DL → ML handoff</summary>
                  <p>
                    {evidence.classical_ml.counts?.train.toLocaleString()}{" "}
                    training examples ·{" "}
                    {evidence.classical_ml.counts?.validation.toLocaleString()}{" "}
                    validation examples ·{" "}
                    {evidence.classical_ml.counts?.test.toLocaleString()}{" "}
                    original test examples. The LSTM and scalers stay frozen. No
                    scenario or uploaded file retrains the head.
                  </p>
                  <div className="s-code-tags">
                    {evidence.classical_ml.latent_values?.map((x) => (
                      <code key={x.name}>
                        {x.name}: {x.value.toFixed(4)}
                      </code>
                    ))}
                  </div>
                  <p>
                    Run: {evidence.classical_ml.run_id}. The test period was
                    used by earlier experiments; it is not a new independent
                    household evaluation.
                  </p>
                </details>
              </>
            )}
            <p className="s-help">
              {evidence?.classical_ml.available
                ? "Real test comparison from the new offline training run. Selection used validation RMSE before this test evaluation; these scores are not recomputed from your scenario."
                : "Authentic saved comparison from the older standalone run. Basic MLP is another neural network; persistence and seasonal methods are references. These are not new evaluation results from your scenario."}
            </p>
            {pending}
            {evidence && (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Saved method</th>
                      <th>MAE (kW)</th>
                      <th>RMSE (kW)</th>
                      <th>R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evidence.comparison.map((x) => (
                      <tr key={x.Model}>
                        <td>{x.Model}</td>
                        <td>{x.MAE_kW.toFixed(3)}</td>
                        <td>{x.RMSE_kW.toFixed(3)}</td>
                        <td>{x.R2.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </>
    );
  if (stage === 3)
    return (
      <>
        {picker}
        <div className="s-ontology-flow">
          <div>
            <small>HOUSEHOLD ENTITY</small>
            <Zap size={27} />
            <h3>{r.Appliance}</h3>
          </div>
          <ArrowRight />
          <div>
            <small>INFERRED CLASSES</small>
            <div className="s-code-tags">
              {r.Ontology_types.map((x) => (
                <code key={x}>{x}</code>
              ))}
            </div>
          </div>
          <ArrowRight />
          <div>
            <small>OWL CANDIDATES</small>
            <h3>{r.OWL_candidates.join(", ") || "No matching head"}</h3>
          </div>
        </div>
        <Notice>
          <strong>Structured knowledge supplies candidates.</strong> Operational
          rules resolve them into the final recommendation. A flexible candidate
          does not override essential protection.
        </Notice>
        <div className="s-rule-cards">
          {boot.ontology.rules.map((x) => (
            <div key={x.id}>
              <Network size={18} />
              <code>{x.id}</code>
              <p>
                {x.classes.join(" AND ")} → {x.decisions.join(", ")}
              </p>
            </div>
          ))}
        </div>
        <details className="s-technical-details">
          <summary>Ontology scope and asserted examples</summary>
          <p>
            SmartElectricityOptimizer.owl: {boot.ontology.classes} classes;{" "}
            {boot.ontology.rules.length} enabled rules. Named subclass closure
            and unary SWRL are supported. This is not a general OWL DL
            consistency reasoner.
          </p>
          <p>
            Asserted example decisions for this identity:{" "}
            {r.OWL_asserted_decisions.join(", ") || "None"}. These remain
            separate from inferred candidates.
          </p>
          <code>{run.ontology_sha256}</code>
        </details>
      </>
    );
  if (stage === 4)
    return (
      <>
        {picker}
        <div className="s-rule-outcome">
          <ShieldCheck size={29} />
          <div>
            <small>ACTUAL RECOMMENDATION</small>
            <h3>{r.Action}</h3>
            <p>{r.Reason}</p>
          </div>
        </div>
        <div className="s-policy-list">
          {[
            [
              "R1-essential",
              "Essential protection",
              "Protect an essential identity.",
            ],
            [
              "R5-empty-home",
              "Occupancy",
              "Stop occupancy-dependent loads in an empty home.",
            ],
            [
              "R2-comfort",
              "Comfort",
              "Protect AC when occupied and temperature is at least 38°C.",
            ],
            [
              "R6-not-requested",
              "Requested state",
              "Leave an unrequested, inactive appliance off.",
            ],
            [
              "R3-priority",
              "Priority & running state",
              "Protect high-priority and already-running appliances.",
            ],
            [
              "R4-forecast-tariff",
              "Flexible jobs",
              "Search allowed times using tariffs and the forecast interval.",
            ],
            [
              "R6-fixed-job",
              "Fixed timing",
              "Use the earliest permitted start for a fixed job.",
            ],
          ].map(([id, title, desc], i) => (
            <div key={id} className={r.Rule_trace.includes(id) ? "fired" : ""}>
              <span>
                {r.Rule_trace.includes(id) ? <Check size={17} /> : i + 1}
              </span>
              <div>
                <strong>{title}</strong>
                <p>{desc}</p>
              </div>
              <code>{id}</code>
              {r.Rule_trace.includes(id) && <b>Fired</b>}
            </div>
          ))}
        </div>
        <p className="s-help">
          Rules are evaluated in this order. The first operational match
          resolves the recommendation. These are executed traces, not a
          simulated animation.
        </p>
      </>
    );
  if (stage === 5)
    return (
      <>
        <div className="s-three-stats">
          <Stat
            label="Explored A* states"
            value={run.plan.nodes.toLocaleString()}
            note="From the actual search"
          />
          <Stat
            label="Scheduled jobs"
            value={run.plan.schedule.length}
            note={`${run.scenario.max_parallel_jobs} at a time permitted`}
          />
          <Stat
            label="Plan status"
            value={run.plan.status.replaceAll("_", " ")}
            note="Under encoded constraints"
          />
        </div>
        <ScheduleView run={run} onAdjust={onAdjust} />
        <details className="s-technical-details">
          <summary>Search objective and cost calculation</summary>
          <p>
            <code>f(n) = g(n) + h(n)</code>. Cost already assigned plus an
            optimistic remaining-cost bound guides A*. Runtime, start/deadline
            windows, concurrency and horizon restrict choices. Forecast
            avoidance applies only to the known 15-minute high-demand target.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Energy</th>
                  <th>Estimated cost</th>
                </tr>
              </thead>
              <tbody>
                {run.plan.schedule.map((x) => (
                  <tr key={x.Appliance}>
                    <td>{x.Appliance}</td>
                    <td>{x.Energy_kWh.toFixed(2)} kWh</td>
                    <td>{money(x.Estimated_cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Feasible earliest-start cost {money(run.plan.baseline_cost)} −
            recommended cost {money(run.plan.estimated_cost)} ={" "}
            {money(run.plan.estimated_saving)} estimated saving. Protected loads
            are outside this comparison.
          </p>
        </details>
      </>
    );
  if (stage === 6)
    return (
      <>
        {pending}
        {evidence && <ShapView evidence={evidence} />}
      </>
    );
  return (
    <>
      <Notice tone="green">
        <strong>The pipeline is complete.</strong> The same forecast, rules and
        scheduler serve the household workspace, with technical evidence exposed
        here.
      </Notice>
      <PlanResult run={run} onAdjust={onAdjust} technical />
      <details className="s-technical-details">
        <summary>Evaluation coverage and limits</summary>
        <p>
          Original full evaluation remains offline. The bundled LSTM’s saved
          test RMSE is 0.6182 kW. Confidence and accuracy for a new household
          are unestablished. Synthetic demonstrations are not accuracy
          evaluations.
        </p>
        <p>
          The current pipeline’s model and evaluation are identified in the forecast stage. No live meter, hardware switching or measured household bill reduction is claimed.
        </p>
      </details>
    </>
  );
}
