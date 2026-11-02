import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  CalendarDays,
  Check,
  Database,
  Download,
  FileClock,
  FileSpreadsheet,
  FlaskConical,
  Home,
  Info,
  Layers3,
  LoaderCircle,
  Play,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import type { Bootstrap, ReadingsReport, Run, SavedRun } from "./types";
import {
  clone,
  day,
  Evidence,
  Inputs,
  MeterUpload,
  money,
  Notice,
  Preferences,
  request,
  time,
} from "./workspace";
import DemoStages, { stages } from "./DemoStages";
import PlanResult from "./PlanResult";
type Mode = "home" | "demo" | "history";
export default function Studio() {
  const [mode, setMode] = useState<Mode>("home"),
    [boot, setBoot] = useState<Bootstrap | null>(null),
    [home, setHome] = useState<Inputs | null>(null),
    [demo, setDemo] = useState<Inputs | null>(null),
    [homeRun, setHomeRun] = useState<Run | null>(null),
    [demoRun, setDemoRun] = useState<Run | null>(null),
    [saved, setSaved] = useState<SavedRun[]>([]);
  const [homeStep, setHomeStep] = useState(0),
    [demoStep, setDemoStep] = useState(0),
    [busy, setBusy] = useState(false),
    [uploadBusy, setUploadBusy] = useState(false),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [homeDirty, setHomeDirty] = useState(false),
    [demoDirty, setDemoDirty] = useState(false),
    [filter, setFilter] = useState("");
  const [evidence, setEvidence] = useState<Evidence | null>(null),
    [evidenceBusy, setEvidenceBusy] = useState(false),
    [evidenceError, setEvidenceError] = useState("");
  const started = useRef(false),
    latestEvidence = useRef(0);
  const stageRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    void (async () => {
      try {
        const [b, h] = await Promise.all([
          request<Bootstrap>("/api/bootstrap"),
          request<SavedRun[]>("/api/history"),
        ]);
        setBoot(b);
        setSaved(h);
        let scenario = b.scenario,
          appliances = b.appliances;
        if (h.length) {
          const r = await request<Run>(`/api/history/${h[0].id}`);
          scenario = r.scenario;
          appliances = r.appliances;
        }
        try {
          const prior = JSON.parse(
            localStorage.getItem("current-household-preferences-v2") || "null",
          );
          if (
            prior?.scenario &&
            Array.isArray(prior.appliances) &&
            prior.appliances.length > 0
          ) {
            scenario = prior.scenario;
            appliances = prior.appliances;
          }
        } catch {
          /* Ignore unavailable or invalid local storage. */
        }
        const initial: Inputs = {
          scenario: clone(scenario),
          appliances: clone(appliances),
          source: "uploaded",
          caseId: b.default_case,
          report: null,
        };
        setHome(initial);
        setDemo({
          ...clone(initial),
          source: "recorded",
          caseId:
            b.cases.find(
              (c) =>
                Number(c.origin.slice(11, 13)) >= 17 &&
                Number(c.origin.slice(11, 13)) < 20,
            )?.id || b.default_case,
        });
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);
  useEffect(() => {
    if (home)
      try {
        localStorage.setItem(
          "current-household-preferences-v2",
          JSON.stringify({
            scenario: home.scenario,
            appliances: home.appliances,
          }),
        );
      } catch {
        /* The workspace still works when local storage is unavailable. */
      }
  }, [home]);
  function change(v: Inputs) {
    if (mode === "demo") {
      setDemo(v);
      setDemoDirty(true);
    } else {
      setHome(v);
      setHomeDirty(true);
    }
  }
  function navigate(m: Mode) {
    if (busy || uploadBusy) return;
    setMode(m);
    setError("");
  }
  function goStep(n: number) {
    setDemoStep(n);
    setTimeout(() => stageRef.current?.focus(), 0);
  }
  async function loadSample() {
    if (!demo) return;
    setUploadBusy(true);
    setError("");
    try {
      const r = await fetch("/api/readings/samples/filled");
      if (!r.ok) throw new Error("Sample could not be loaded.");
      const report = await request<ReadingsReport>(
        "/api/readings?timezone=UTC",
        {
          method: "POST",
          headers: {
            "Content-Type": "text/csv",
            "X-File-Name": "SYNTHETIC-electricity-readings-8-days.csv",
          },
          body: await r.blob(),
        },
      );
      setDemo({ ...demo, source: "sample", report });
      setDemoDirty(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploadBusy(false);
    }
  }
  async function explain(r: Run) {
    const seq = ++latestEvidence.current;
    setEvidence(null);
    setEvidenceBusy(true);
    setEvidenceError("");
    try {
      const e = await request<Evidence>(`/api/demo/${r.id}/evidence`, {
        method: "POST",
      });
      if (seq === latestEvidence.current) setEvidence(e);
    } catch (e) {
      if (seq === latestEvidence.current)
        setEvidenceError((e as Error).message);
    } finally {
      if (seq === latestEvidence.current) setEvidenceBusy(false);
    }
  }
  async function analyze(event?: React.FormEvent) {
    event?.preventDefault();
    const v = mode === "demo" ? demo : home;
    if (!v) return;
    setError("");
    setBusy(true);
    try {
      const source =
        v.source === "recorded"
          ? { case_id: v.caseId }
          : {
              upload_id: v.report?.id,
              sample_forecast: mode === "demo" && !!v.report?.sample_ready,
            };
      const r = await request<Run>("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...source,
          scenario: v.scenario,
          appliances: v.appliances,
        }),
      });
      if (mode === "demo") {
        setDemoRun(r);
        setDemoDirty(false);
        goStep(1);
        void explain(r);
      } else {
        setHomeRun(r);
        setHomeDirty(false);
        setHomeStep(2);
      }
      setSaved(await request<SavedRun[]>("/api/history"));
    } catch (e) {
      setError((e as Error).message);
      if (v.report?.id) {
        try {
          const report = await request<ReadingsReport>(
            `/api/readings/${v.report.id}`,
          );
          const update = (old: Inputs | null) =>
            old && old.report?.id === v.report?.id ? { ...old, report } : old;
          if (mode === "demo") setDemo(update);
          else setHome(update);
        } catch {
          /* Keep the original actionable analysis error. */
        }
      }
    } finally {
      setBusy(false);
    }
  }
  function preset(name: string) {
    if (!demo || !boot) return;
    const scenario = clone(boot.scenario),
      appliances = clone(boot.appliances);
    if (name === "Comfort first") {
      scenario.occupancy = 4;
      scenario.temperature_c = 40;
    }
    if (name === "Empty home") {
      scenario.occupancy = 0;
      scenario.temperature_c = 30;
    }
    if (name === "Tight deadline") appliances[2].deadline_min = 30;
    setDemo({ ...demo, scenario, appliances });
    setDemoDirty(true);
  }
  async function restore(id: string) {
    setBusy(true);
    setError("");
    try {
      const r = await request<Run>(`/api/history/${id}`);
      const isDemo =
        r.forecast.is_sample || r.source_mode !== "uploaded_readings";
      const v: Inputs = {
        scenario: r.scenario,
        appliances: r.appliances,
        source: r.forecast.is_sample
          ? "sample"
          : r.upload_id
            ? "uploaded"
            : "recorded",
        caseId: r.case_id,
        report: r.upload_id
          ? await request<ReadingsReport>(`/api/readings/${r.upload_id}`)
          : null,
      };
      if (isDemo) {
        setDemo(v);
        setDemoRun(r);
        setDemoDirty(false);
        setDemoStep(7);
        setMode("demo");
        setEvidence(null);
        latestEvidence.current++;
        setEvidenceBusy(false);
      } else {
        setHome(v);
        setHomeRun(r);
        setHomeDirty(false);
        setHomeStep(2);
        setMode("home");
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const current = mode === "demo" ? demo : home;
  const ready =
    !!current &&
    (current.source === "recorded" ||
      !!current.report?.can_analyze ||
      (mode === "demo" && !!current.report?.sample_ready));
  const disabled = busy || uploadBusy;
  const stage = stages[demoStep];
  return (
    <div className="studio">
      <a className="s-skip" href="#workspace">
        Skip to workspace
      </a>
      <header className="s-header">
        <a
          href="#"
          className="s-brand"
          onClick={(e) => {
            e.preventDefault();
            navigate("home");
          }}
        >
          <span>
            <Zap size={23} fill="currentColor" />
          </span>
          current<span className="s-brand-period">.</span>
        </a>
        <nav aria-label="Main navigation">
          {(
            [
              { id: "home", label: "My household", icon: Home },
              { id: "demo", label: "Guided demo", icon: FlaskConical },
              { id: "history", label: "History", icon: FileClock },
            ] as const
          ).map((x) => (
            <button
              key={x.id}
              disabled={disabled}
              aria-current={mode === x.id ? "page" : undefined}
              className={mode === x.id ? "active" : ""}
              onClick={() => navigate(x.id)}
            >
              <x.icon size={16} />
              {x.label}
            </button>
          ))}
        </nav>
        <span className="s-local">
          <i />
          Local workspace
        </span>
      </header>
      <main id="workspace" className="s-main">
        {error && (
          <div className="s-error" role="alert">
            <Info size={19} />
            <div>
              <strong>Something needs attention</strong>
              <p>{error}</p>
            </div>
            <button aria-label="Dismiss error" onClick={() => setError("")}>
              <X size={18} />
            </button>
          </div>
        )}
        {loading ? (
          <div className="s-loading">
            <LoaderCircle className="spin" />
            Opening your electricity workspace…
          </div>
        ) : !boot || !home || !demo ? (
          <Notice tone="orange">
            The local service could not be loaded. Refresh when the server is
            available.
          </Notice>
        ) : (
          <>
            {mode === "home" && (
              <>
                <section className="s-home-hero">
                  <div>
                    <span className="s-eyebrow">
                      <i />A LITTLE PLANNING GOES A LONG WAY
                    </span>
                    <h1>
                      A clearer plan.
                      <br />
                      <em>A lighter bill.</em>
                    </h1>
                    <p>
                      Understand your electricity use and find a better time to
                      run your appliances. One simple step at a time.
                    </p>
                    <div className="s-hero-actions">
                      <button
                        className="s-button"
                        onClick={() => {
                          setHomeStep(0);
                          document
                            .getElementById("home-setup")
                            ?.scrollIntoView({ behavior: "smooth" });
                        }}
                      >
                        Plan my electricity <ArrowRight size={17} />
                      </button>
                      <button
                        className="s-text"
                        onClick={() => navigate("demo")}
                      >
                        <Play size={15} />
                        See how it works
                      </button>
                    </div>
                    <div className="s-hero-trust">
                      <ShieldCheck size={15} />
                      Your priorities protected<span>·</span>No appliance
                      switching
                    </div>
                  </div>
                  <div className="s-hero-art" aria-hidden="true">
                    <div className="s-art-orbit orbit-a" />
                    <div className="s-art-orbit orbit-b" />
                    <div className="s-art-house">
                      <Home size={62} strokeWidth={1.3} />
                      <span>
                        <i />A home in balance
                      </span>
                    </div>
                    <div className="s-art-chip top">
                      <ShieldCheck size={18} />
                      <div>
                        <b>Comfort comes first</b>
                        <small>Keep essentials protected</small>
                      </div>
                    </div>
                    <div className="s-art-chip bottom">
                      <CalendarDays size={19} />
                      <div>
                        <b>Find a better time</b>
                        <small>A plan around your day</small>
                      </div>
                      <span className="s-mini-bars">
                        <i />
                        <i />
                        <i />
                        <i />
                        <i />
                      </span>
                    </div>
                  </div>
                </section>
                <div className="s-home-layout">
                  <aside className="s-card s-get-started">
                    <span className="s-overline">YOUR NEXT STEP</span>
                    <h2>Make it work for your home.</h2>
                    <p>
                      Add your readings and preferences. We’ll turn them into a
                      clear, practical appliance plan.
                    </p>
                    <ol>
                      {[
                        "Add electricity readings",
                        "Set household preferences",
                        "Review your plan",
                      ].map((x, i) => (
                        <li
                          key={x}
                          className={
                            homeStep === i
                              ? "active"
                              : (i === 0 && home.report?.can_analyze) ||
                                  (i === 1 && homeRun && !homeDirty)
                                ? "done"
                                : ""
                          }
                        >
                          <button
                            disabled={i === 2 && !homeRun}
                            onClick={() => setHomeStep(i)}
                          >
                            <span>
                              {(i === 0 && home.report?.can_analyze) ||
                              (i === 1 && homeRun && !homeDirty) ? (
                                <Check size={15} />
                              ) : (
                                i + 1
                              )}
                            </span>
                            {x}
                          </button>
                        </li>
                      ))}
                    </ol>
                    <div className="s-demo-invite">
                      <FlaskConical size={21} />
                      <strong>Just exploring?</strong>
                      <p>
                        Try the guided demo with recorded examples or sample
                        readings.
                      </p>
                      <button
                        className="s-text"
                        onClick={() => navigate("demo")}
                      >
                        Open guided demo <ArrowRight size={14} />
                      </button>
                    </div>
                  </aside>
                  <section id="home-setup" className="s-home-content">
                    {homeStep === 0 && (
                      <div className="s-card">
                        <div className="s-section-title">
                          <div>
                            <span className="s-overline">
                              STEP 01 / READINGS
                            </span>
                            <h2>Start with your electricity data.</h2>
                          </div>
                          <Database size={24} />
                        </div>
                        <p className="s-intro">
                          A recent meter export lets the estimate follow your
                          home’s actual usage.
                        </p>
                        <MeterUpload
                          value={home}
                          onChange={change}
                          onBusy={setUploadBusy}
                          onError={setError}
                        />
                        <div className="s-card-footer">
                          <span>
                            {home.report?.can_analyze
                              ? "Readings checked and ready"
                              : "Check your readings to continue"}
                          </span>
                          <button
                            className="s-button"
                            disabled={!home.report?.can_analyze || disabled}
                            onClick={() => setHomeStep(1)}
                          >
                            Continue <ArrowRight size={16} />
                          </button>
                        </div>
                      </div>
                    )}
                    {homeStep === 1 && (
                      <form className="s-card" onSubmit={analyze}>
                        <div className="s-section-title">
                          <div>
                            <span className="s-overline">
                              STEP 02 / PREFERENCES
                            </span>
                            <h2>A little context. A better plan.</h2>
                          </div>
                          <Settings2 size={24} />
                        </div>
                        <fieldset disabled={disabled}>
                          <Preferences value={home} onChange={change} />
                        </fieldset>
                        {!home.report?.can_analyze && (
                          <Notice>
                            Add and check recent household readings before
                            generating a plan. You can save your preferences
                            here first, or explore sample data in Guided demo.
                          </Notice>
                        )}
                        <div className="s-card-footer">
                          <button
                            type="button"
                            className="s-text"
                            onClick={() => setHomeStep(0)}
                          >
                            <ArrowLeft size={15} />
                            Readings
                          </button>
                          <button
                            className="s-button"
                            disabled={!ready || disabled}
                          >
                            {busy ? (
                              <LoaderCircle className="spin" size={16} />
                            ) : (
                              <Sparkles size={16} />
                            )}
                            Generate my plan
                          </button>
                        </div>
                      </form>
                    )}
                    {homeStep === 2 && homeRun && (
                      <>
                        <div className="s-inline-between s-result-heading">
                          <div>
                            <span className="s-overline">
                              STEP 03 / YOUR PLAN
                            </span>
                            <h2>A few clear next steps.</h2>
                          </div>
                          <button
                            className="s-button secondary"
                            onClick={() => setHomeStep(1)}
                          >
                            <Settings2 size={15} />
                            Edit preferences
                          </button>
                        </div>
                        {homeDirty && (
                          <Notice tone="orange">
                            Preferences changed. This result belongs to the
                            saved analysis; generate a new plan to apply edits.
                          </Notice>
                        )}
                        <PlanResult
                          run={homeRun}
                          onAdjust={() => setHomeStep(1)}
                        />
                      </>
                    )}
                  </section>
                </div>
              </>
            )}
            {mode === "demo" && (
              <>
                <section className="s-demo-heading">
                  <div>
                    <span className="s-eyebrow">
                      <FlaskConical size={15} />
                      THE PROJECT, MADE VISIBLE
                    </span>
                    <h1>
                      Follow the intelligence.
                      <br />
                      <em>From readings to a reason.</em>
                    </h1>
                    <p>
                      Set a scenario, run the real pipeline, then inspect each
                      technology and the evidence behind the result.
                    </p>
                  </div>
                  <div className="s-demo-caption">
                    <Layers3 size={25} />
                    <strong>8 connected stages</strong>
                    <span>Actual outputs. Inspectable evidence.</span>
                    <small>Forecasting · knowledge · search · SHAP</small>
                  </div>
                </section>
                <div className="s-demo-layout">
                  <aside className="s-stage-rail">
                    <span className="s-overline">GUIDED WALKTHROUGH</span>
                    {stages.map((s, i) => (
                      <button
                        key={s.title}
                        aria-current={demoStep === i ? "step" : undefined}
                        className={demoStep === i ? "active" : ""}
                        onClick={() => goStep(i)}
                      >
                        <span className="s-step-icon">
                          <s.icon size={18} />
                        </span>
                        <span>
                          <strong>{s.title}</strong>
                          <small>{s.sub}</small>
                        </span>
                        <span className="s-step-no">
                          {String(i + 1).padStart(2, "0")}
                        </span>
                      </button>
                    ))}
                    <div className="s-rail-note">
                      <ShieldCheck size={16} />
                      <span>
                        Run once, inspect each stage. No training happens on a
                        click.
                      </span>
                    </div>
                  </aside>
                  <section className="s-stage-card s-card">
                    <div className="s-stage-heading">
                      <span className="s-stage-count">
                        {String(demoStep + 1).padStart(2, "0")}
                        <small>/ 08</small>
                      </span>
                      <div>
                        <span className="s-overline">{stage.sub}</span>
                        <h2 ref={stageRef} tabIndex={-1}>
                          {stage.title}
                        </h2>
                      </div>
                      <stage.icon size={25} />
                    </div>
                    {demoDirty && demoRun && demoStep > 0 && (
                      <Notice tone="orange">
                        Inputs changed. You’re inspecting the previous run.
                        Return to Set the scene to run the updated scenario.
                      </Notice>
                    )}
                    {demoStep === 0 ? (
                      <form onSubmit={analyze}>
                        <p className="s-intro">
                          Choose electricity history and household values. The
                          original rules and scheduler will turn them into a
                          plan.
                        </p>
                        <label className="s-field">
                          <span>Electricity source</span>
                          <select
                            disabled={disabled}
                            value={demo.source}
                            onChange={(e) =>
                              change({
                                ...demo,
                                source: e.target.value as Inputs["source"],
                                report: null,
                              })
                            }
                          >
                            <option value="recorded">
                              Recorded household example
                            </option>
                            <option value="sample">
                              Filled eight-day sample
                            </option>
                            <option value="uploaded">Upload a CSV</option>
                          </select>
                        </label>
                        {demo.source === "recorded" && (
                          <label className="s-field">
                            <span>Recorded history · original dates</span>
                            <select
                              disabled={disabled}
                              value={demo.caseId}
                              onChange={(e) =>
                                change({ ...demo, caseId: e.target.value })
                              }
                            >
                              {boot.cases.map((c) => (
                                <option key={c.id} value={c.id}>
                                  {day(c.origin)} · {time(c.origin)}
                                </option>
                              ))}
                            </select>
                          </label>
                        )}
                        {demo.source === "sample" && (
                          <div className="s-sample-choice">
                            <FileSpreadsheet size={29} />
                            <div>
                              <strong>Eight days, ready to explore.</strong>
                              <p>
                                11,520 synthetic minute readings · 1–8 Jan 2026
                                · UTC
                              </p>
                              {demo.report && (
                                <small>
                                  Loaded · target {demo.report.target_end} UTC
                                </small>
                              )}
                            </div>
                            <button
                              type="button"
                              className="s-button secondary"
                              disabled={disabled}
                              onClick={() => void loadSample()}
                            >
                              {uploadBusy ? (
                                <LoaderCircle className="spin" size={15} />
                              ) : (
                                <Download size={15} />
                              )}{" "}
                              {demo.report ? "Reload sample" : "Load sample"}
                            </button>
                          </div>
                        )}
                        {demo.source === "uploaded" && (
                          <MeterUpload
                            value={demo}
                            demoMode
                            onChange={change}
                            onBusy={setUploadBusy}
                            onError={setError}
                          />
                        )}
                        <div className="s-presets">
                          <span>Explore a scenario</span>
                          <div>
                            {[
                              "Tariff planning",
                              "Comfort first",
                              "Empty home",
                              "Tight deadline",
                            ].map((x) => (
                              <button
                                type="button"
                                disabled={disabled}
                                key={x}
                                onClick={() => preset(x)}
                              >
                                {x}
                              </button>
                            ))}
                          </div>
                          <button
                            type="button"
                            className="s-text"
                            disabled={disabled}
                            onClick={() => {
                              setDemo({
                                ...demo,
                                scenario: clone(home.scenario),
                                appliances: clone(home.appliances),
                              });
                              setDemoDirty(true);
                            }}
                          >
                            Use my household preferences
                          </button>
                          <small>
                            Presets change household preferences only. Readings
                            and predictions are not fabricated.
                          </small>
                        </div>
                        <fieldset disabled={disabled}>
                          <Preferences value={demo} onChange={change} />
                        </fieldset>
                        <div className="s-card-footer">
                          <span>
                            {demo.source === "sample"
                              ? "Synthetic sample · original dates"
                              : "Household values guide rules and costs"}
                          </span>
                          <button
                            className="s-button"
                            disabled={!ready || disabled}
                          >
                            {busy ? (
                              <LoaderCircle className="spin" size={17} />
                            ) : (
                              <Play size={16} />
                            )}{" "}
                            {busy
                              ? "Running the pipeline…"
                              : "Run demonstration"}
                          </button>
                        </div>
                      </form>
                    ) : demoRun ? (
                      <DemoStages
                        stage={demoStep}
                        run={demoRun}
                        boot={boot}
                        evidence={evidence}
                        evidenceBusy={evidenceBusy}
                        evidenceError={evidenceError}
                        onEvidence={() => void explain(demoRun)}
                        onAdjust={() => goStep(0)}
                      />
                    ) : (
                      <div className="s-empty">
                        <stage.icon size={39} />
                        <h3>Start with a real run.</h3>
                        <p>
                          Set the scenario and run the demonstration to inspect
                          actual outputs here.
                        </p>
                        <button className="s-button" onClick={() => goStep(0)}>
                          Set the scene <ArrowLeft size={16} />
                        </button>
                      </div>
                    )}
                    {demoStep > 0 && (
                      <div className="s-stage-footer">
                        <button
                          className="s-text"
                          onClick={() => goStep(demoStep - 1)}
                        >
                          <ArrowLeft size={16} />
                          Previous stage
                        </button>
                        <span>{demoStep + 1} of 8</span>
                        {demoStep < 7 ? (
                          <button
                            className="s-button"
                            onClick={() => goStep(demoStep + 1)}
                          >
                            Next stage <ArrowRight size={16} />
                          </button>
                        ) : (
                          <button
                            className="s-button secondary"
                            onClick={() => goStep(0)}
                          >
                            <RefreshCw size={15} />
                            Change scenario
                          </button>
                        )}
                      </div>
                    )}
                  </section>
                </div>
              </>
            )}
            {mode === "history" && (
              <>
                <section className="s-history-heading">
                  <span className="s-eyebrow">
                    <FileClock size={15} />
                    YOUR SAVED ANALYSES
                  </span>
                  <h1>
                    Pick up where
                    <br />
                    <em>you left off.</em>
                  </h1>
                  <p>Original inputs, dates and results, preserved together.</p>
                </section>
                <section className="s-card">
                  <div className="s-inline-between">
                    <h2>
                      Analysis history{" "}
                      <span className="s-tag">{saved.length}</span>
                    </h2>
                    <label className="s-field">
                      <span className="sr-only">Search history</span>
                      <input
                        placeholder="Search date, source or result…"
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                      />
                    </label>
                  </div>
                  {!saved.length ? (
                    <div className="s-empty">
                      <FileClock size={35} />
                      <h3>Your first plan starts here.</h3>
                      <p>Completed analyses appear here automatically.</p>
                      <button
                        className="s-button"
                        onClick={() => navigate("home")}
                      >
                        Plan my electricity
                      </button>
                    </div>
                  ) : (
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Saved analysis</th>
                            <th>Reading date & source</th>
                            <th>Forecast</th>
                            <th>Job saving</th>
                            <th>Status</th>
                            <th />
                          </tr>
                        </thead>
                        <tbody>
                          {saved
                            .filter((r) =>
                              `${r.forecast.origin} ${r.forecast.is_sample ? "synthetic sample" : r.source_mode} ${r.plan.status}`
                                .toLowerCase()
                                .includes(filter.toLowerCase()),
                            )
                            .map((r) => (
                              <tr key={r.id}>
                                <td>
                                  {new Date(r.created_at).toLocaleDateString(
                                    "en-IN",
                                  )}
                                  <small>
                                    {new Date(r.created_at).toLocaleTimeString(
                                      "en-IN",
                                      { hour: "2-digit", minute: "2-digit" },
                                    )}
                                  </small>
                                </td>
                                <td>
                                  {day(r.forecast.origin)}
                                  <small>
                                    {r.forecast.is_sample
                                      ? "Synthetic sample"
                                      : r.source_mode === "uploaded_readings"
                                        ? "Household upload"
                                        : "Recorded example"}
                                  </small>
                                </td>
                                <td>{r.forecast.forecast_kw.toFixed(3)} kW</td>
                                <td>{money(r.plan.estimated_saving)}</td>
                                <td>
                                  <span className="s-tag">
                                    {r.plan.status.replaceAll("_", " ")}
                                  </span>
                                </td>
                                <td>
                                  <button
                                    className="s-text"
                                    disabled={busy}
                                    onClick={() => void restore(r.id)}
                                  >
                                    Open <ArrowRight size={14} />
                                  </button>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>
              </>
            )}
          </>
        )}
      </main>
      <footer className="s-footer">
        <span>
          <Zap size={15} />
          current. <span>Thoughtful electricity planning.</span>
        </span>
        <details>
          <summary>Data & limitations</summary>
          <p>
            Local advisory software; no appliance control or utility connection.
            Estimates cover one 15-minute interval one hour after the latest
            complete reading. Household settings guide decisions, not simulated
            aggregate demand. Confidence and new-household accuracy are
            unestablished. Files/history stay in the local workspace; sample
            results retain synthetic identity. Job savings are scenario
            estimates.
          </p>
        </details>
        <span>Decisions stay with you.</span>
      </footer>
    </div>
  );
}
