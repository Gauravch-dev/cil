import {
  Activity,
  ChevronDown,
  Download,
  ShieldCheck,
  Zap,
} from "lucide-react";
import type { Run } from "./types";
import { DemandChart, ScheduleView } from "./ResultComponents";
import { day, money, Notice, time } from "./workspace";

export default function PlanResult({
  run,
  onAdjust,
  technical = false,
}: {
  run: Run;
  onAdjust: () => void;
  technical?: boolean;
}) {
  return (
    <div className="s-result-stack">
      {run.forecast.is_sample && (
        <Notice tone="orange">
          <strong>Sample-date result.</strong> Synthetic readings at their
          original dates; this is not your actual current household use.
        </Notice>
      )}
      <div className="s-outcome-grid">
        <div className="s-forecast">
          <span className="s-overline">
            {run.forecast.is_sample
              ? "SAMPLE DEMAND ESTIMATE"
              : "YOUR DEMAND ESTIMATE"}
          </span>
          <div className="s-big-number">
            {run.forecast.forecast_kw.toFixed(3)}
            <span>kW</span>
          </div>
          <p>
            {day(run.forecast.target_end)} ·{" "}
            {time(run.forecast.target_interval_start)}–
            {time(run.forecast.target_end)}{" "}
            {run.uploaded_readings?.timezone || "recorded local clock"}
          </p>
          <span className="s-forecast-status">
            <Activity size={14} />
            {run.forecast.forecast_kw >= run.forecast.threshold_kw
              ? "High demand"
              : "Normal demand"}
          </span>
          <small>
            One 15-minute interval, ending one hour after the latest reading.
            Confidence is not quantified.
          </small>
        </div>
        <div className="s-cost-card">
          <span className="s-overline">YOUR SCHEDULED JOBS</span>
          <strong>{money(run.plan.estimated_cost)}</strong>
          <p>Estimated cost of the recommended plan</p>
          <div>
            <span>Earliest feasible plan</span>
            <b>{money(run.plan.baseline_cost)}</b>
          </div>
          <div>
            <span>Possible saving</span>
            <b className="s-green">{money(run.plan.estimated_saving)}</b>
          </div>
          <small>
            Scenario estimate for scheduled jobs; actual bill savings are not
            measured.
          </small>
        </div>
      </div>
      <section className="s-card">
        <div className="s-section-title">
          <div>
            <span className="s-overline">WHAT TO DO NEXT</span>
            <h2>Your appliance decisions</h2>
          </div>
          <ShieldCheck size={23} />
        </div>
        <div className="s-decisions">
          {run.recommendations.map((r) => (
            <details key={r.Appliance}>
              <summary>
                <span className={`s-decision-dot ${r.Kind}`}>
                  {r.Kind === "protected" ? (
                    <ShieldCheck size={18} />
                  ) : (
                    <Zap size={18} />
                  )}
                </span>
                <span>
                  <strong>{r.Appliance}</strong>
                  <small>
                    {r.Action}
                    {r.Suggested_start && ` · ${time(r.Suggested_start)}`}
                  </small>
                </span>
                <ChevronDown size={16} />
              </summary>
              <p>{r.Reason}</p>
              {technical && (
                <div className="s-code-tags">
                  {r.Rule_trace.map((x) => (
                    <code key={x}>{x}</code>
                  ))}
                </div>
              )}
            </details>
          ))}
        </div>
      </section>
      <ScheduleView run={run} onAdjust={onAdjust} />
      <details className="s-card s-result-details">
        <summary>
          Explore the electricity readings <ChevronDown size={17} />
        </summary>
        <DemandChart run={run} />
      </details>
      <div className="s-export-row">
        <span>Saved {new Date(run.created_at).toLocaleString("en-IN")}</span>
        <a href={`/api/history/${run.id}/export/json`} download>
          <Download size={14} />
          Analysis JSON
        </a>
        <a href={`/api/history/${run.id}/export/schedule`} download>
          Schedule CSV
        </a>
        <a href={`/api/history/${run.id}/export/recommendations`} download>
          Recommendations
        </a>
      </div>
    </div>
  );
}
