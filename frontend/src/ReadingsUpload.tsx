import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  Clock3,
  Download,
  ChevronDown,
  FileSpreadsheet,
  Info,
  LoaderCircle,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import type { ReadingsReport, ReadingsSchema } from "./types";

type Props = {
  templatesOpen: boolean;
  onTemplatesOpen: (open: boolean) => void;
  schema: ReadingsSchema;
  serviceAvailable: boolean;
  report: ReadingsReport | null;
  disabled: boolean;
  uploading: boolean;
  onReport: (report: ReadingsReport | null) => void;
  onBusyChange: (busy: boolean) => void;
};
const stamp = (s: string | null) =>
  s
    ? new Date(s.replace(" ", "T")).toLocaleString("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Unavailable";

export default function ReadingsUpload({
  templatesOpen,
  onTemplatesOpen,
  schema,
  serviceAvailable,
  report,
  disabled,
  uploading,
  onReport,
  onBusyChange,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [zone, setZone] = useState(() => {
    const detected =
      Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Kolkata";
    return detected === "Asia/Calcutta" ? "Asia/Kolkata" : detected;
  });
  const [error, setError] = useState("");
  const [loadingSample, setLoadingSample] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const templatesRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (templatesOpen) templatesRef.current?.focus();
  }, [templatesOpen]);
  async function check(selectedFile = file, readingZone = zone) {
    if (!selectedFile) return;
    setError("");
    onReport(null);
    onBusyChange(true);
    try {
      const res = await fetch(
        `/api/readings?timezone=${encodeURIComponent(readingZone)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "text/csv",
            "X-File-Name": encodeURIComponent(selectedFile.name),
          },
          body: selectedFile,
        },
      );
      const body = await res.json();
      if (!res.ok)
        throw new Error(
          typeof body.detail === "string"
            ? body.detail
            : "The file could not be checked. Please try again.",
        );
      onReport(body);
    } catch (e) {
      setError(
        e instanceof TypeError
          ? "The local service could not be reached. Please check that the app is running and try again."
          : (e as Error).message ||
              "The file could not be checked. Please try again.",
      );
    } finally {
      onBusyChange(false);
    }
  }
  async function useFilledSample() {
    setError("");
    setLoadingSample(true);
    onBusyChange(true);
    try {
      const response = await fetch("/api/readings/samples/filled");
      if (
        !response.ok ||
        !response.headers.get("content-type")?.startsWith("text/csv")
      )
        throw new Error(
          "The sample could not be loaded. Refresh the page and try again.",
        );
      const sample = new File(
        [await response.blob()],
        "SYNTHETIC-electricity-readings-8-days.csv",
        { type: "text/csv" },
      );
      setFile(sample);
      setZone("UTC");
      if (input.current) input.current.value = "";
      await check(sample, "UTC");
    } catch (e) {
      setError(
        (e as Error).message || "The sample could not be loaded. Try again.",
      );
    } finally {
      setLoadingSample(false);
      onBusyChange(false);
    }
  }
  return (
    <section className="panel readings-panel" aria-labelledby="readings-title">
      <div className="readings-heading">
        <span className="upload-symbol">
          <Upload size={23} />
        </span>
        <div>
          <span className="eyebrow">YOUR DATA. YOUR HOUSEHOLD.</span>
          <h2 id="readings-title">Use my electricity readings</h2>
          <p>
            Upload a meter export to prepare a current-date demand estimate. Its
            date follows the latest usable reading in your file.
          </p>
        </div>
        <button
          className="secondary-button"
          aria-expanded={templatesOpen}
          aria-controls="sample-templates"
          onClick={() => onTemplatesOpen(!templatesOpen)}
        >
          <Download size={16} />
          Sample templates <ChevronDown size={14} />
        </button>
      </div>
      {templatesOpen && (
        <div
          id="sample-templates"
          className="sample-templates"
          ref={templatesRef}
          tabIndex={-1}
          aria-labelledby="templates-heading"
        >
          <div className="templates-title">
            <div>
              <h3 id="templates-heading">Choose a template to download</h3>
              <p>
                Start with the blank file for your readings, or explore a filled
                example to understand the format.
              </p>
            </div>
            <button
              className="icon-button"
              aria-label="Close sample templates"
              onClick={() => onTemplatesOpen(false)}
            >
              <X size={17} />
            </button>
          </div>
          <div className="template-options">
            <a href={schema.template_url} download>
              <span className="template-file-icon">
                <FileSpreadsheet size={22} />
              </span>
              <div>
                <strong>Blank CSV template</strong>
                <p>Required columns, ready for your actual meter readings.</p>
                <small>CSV · header only</small>
              </div>
              <Download size={17} />
            </a>
            <a href="/api/readings/samples/filled" download>
              <span className="template-file-icon sample">
                <FileSpreadsheet size={22} />
              </span>
              <div>
                <strong>Filled sample · 8 days</strong>
                <p>
                  11,520 synthetic minute readings showing the expected format.
                </p>
                <small>CSV · synthetic example · UTC · 1–8 Jan 2026</small>
              </div>
              <Download size={17} />
            </a>
            <a href="/api/readings/samples/guide" download>
              <span className="template-file-icon">
                <Info size={22} />
              </span>
              <div>
                <strong>Column and units guide</strong>
                <p>
                  Timestamp format, measurement units and how to fill your file.
                </p>
                <small>TXT · quick reference</small>
              </div>
              <Download size={17} />
            </a>
          </div>
          <button
            className="secondary-button use-sample-button"
            disabled={disabled || uploading}
            onClick={() => void useFilledSample()}
          >
            {loadingSample ? (
              <LoaderCircle size={17} className="spin" />
            ) : (
              <FileSpreadsheet size={17} />
            )}
            Use filled sample<span>Load and check it directly</span>
          </button>
          <p className="synthetic-note">
            <Info size={16} />
            <span>
              The filled sample is invented example data, not a household
              measurement. It is marked as synthetic and cannot generate a
              current household forecast. You can generate a sample forecast
              using its original dates. Use your own readings in the blank template for
              a real estimate.
            </span>
          </p>
        </div>
      )}
      <ol className="upload-steps">
        <li>
          <b>1</b>Export at least eight days of minute readings
        </li>
        <li>
          <b>2</b>Check the file and its timezone
        </li>
        <li>
          <b>3</b>Generate an estimate when all checks pass
        </li>
      </ol>
      {!serviceAvailable && (
        <div className="upload-service-note">
          <Info size={18} />
          <p>
            <strong>File checks are available.</strong> Current-date estimates
            need a compatible forecasting connection, which is not configured in
            this workspace yet. You can check your file now; no forecast or
            savings will be generated until that connection is ready.
          </p>
        </div>
      )}
      {serviceAvailable && (
        <div className="upload-ready">
          <CheckCircle2 size={21} />
          <p>
            <strong>Forecasting is ready.</strong> Upload your actual meter
            readings, check the file, then select <strong>Analyze my energy</strong>.
            The estimate covers a 15-minute interval one hour after your latest
            complete reading. Accuracy for your household has not been established.
          </p>
        </div>
      )}
      <div className="upload-controls">
        <label className="file-input-label">
          <span>Electricity readings CSV</span>
          <input
            ref={input}
            type="file"
            accept=".csv,text/csv"
            disabled={disabled || uploading}
            onChange={(e) => {
              const chosen = e.target.files?.[0] || null;
              setError("");
              onReport(null);
              if (chosen && chosen.size > schema.maximum_bytes) {
                setFile(null);
                e.target.value = "";
                setError(
                  "Choose a CSV no larger than 10 MB. Export the most recent eight to 60 days.",
                );
                return;
              }
              setFile(chosen);
            }}
          />
          <small>CSV · maximum 10 MB · seven electrical measurements</small>
          {file && (
            <small className="selected-file-name">Selected: {file.name}</small>
          )}
        </label>
        <label className="field timezone-field">
          <span>Timezone used in the CSV</span>
          <select
            value={zone}
            disabled={disabled || uploading}
            onChange={(e) => {
              setZone(e.target.value);
              onReport(null);
            }}
          >
            {Array.from(
              new Set([
                zone,
                "Asia/Kolkata",
                "UTC",
                "Europe/Paris",
                "Europe/London",
                "America/New_York",
                "America/Los_Angeles",
                "Asia/Singapore",
                "Australia/Sydney",
              ]),
            ).map((z) => (
              <option key={z} value={z}>
                {z === "Asia/Calcutta" || z === "Asia/Kolkata"
                  ? "India · Asia/Kolkata"
                  : z}
              </option>
            ))}
          </select>
        </label>
        <button
          className="primary-button"
          disabled={!file || disabled || uploading}
          onClick={() => void check()}
        >
          {uploading ? (
            <LoaderCircle size={17} className="spin" />
          ) : (
            <ShieldCheck size={17} />
          )}{" "}
          {uploading ? "Checking your readings…" : "Check my readings"}
        </button>
      </div>
      <p className="upload-privacy">
        <ShieldCheck size={14} />
        Files are stored in this local workspace. No utility account connection
        or appliance control is performed.
      </p>
      {error && (
        <div className="error-banner" role="alert">
          <Info size={18} />
          <div>
            <strong>The file needs attention</strong>
            <p>{error}</p>
            <p>No forecast was generated from this file.</p>
            <button
              className="text-button"
              onClick={() => onTemplatesOpen(true)}
            >
              Open sample templates <ArrowRight size={14} />
            </button>
          </div>
        </div>
      )}
      {uploading && (
        <p className="upload-progress" role="status">
          Checking columns, timestamps, recent measurements and the forecasting
          connection…
        </p>
      )}
      {report && (
        <div className="readings-report" role="status" aria-live="polite">
          <div className="report-title">
            <div>
              <FileSpreadsheet size={20} />
              <strong>{report.file_name}</strong>
            </div>
            <span className={`pill ${report.can_analyze ? "green" : "orange"}`}>
              {report.can_analyze
                ? "Ready to analyze"
                : report.sample_ready
                  ? "Ready for a sample forecast"
                : report.is_sample
                  ? "Sample loaded · format checked"
                  : report.data_ready
                    ? "Readings checked · estimate unavailable"
                    : "Readings need attention"}
            </span>
          </div>
          <div className="readings-stats">
            <div>
              <span>READINGS</span>
              <strong>{report.row_count.toLocaleString()}</strong>
              <small>{report.days_covered.toFixed(2)} days supplied</small>
            </div>
            <div>
              <span>FILE DATE RANGE</span>
              <strong>{stamp(report.start)}</strong>
              <small>to {stamp(report.end)}</small>
            </div>
            <div>
              <span>LATEST USABLE INTERVAL END</span>
              <strong>{stamp(report.origin)}</strong>
              <small>
                {report.timezone}
                {report.age_minutes != null
                  ? ` · ${Math.round(report.age_minutes)} minutes old when checked`
                  : ""}
              </small>
            </div>
          </div>
          <p className="readings-checked-at">
            Checked {new Date(report.checked_at).toLocaleString("en-IN")}.
            Freshness is checked again before every estimate.
          </p>
          <div className="readings-checks">
            {report.checks.map((check) => (
              <div key={check.code} className={`reading-check ${check.status}`}>
                <span>
                  {check.status === "pass" ? (
                    <CheckCircle2 size={19} />
                  ) : (
                    <Info size={19} />
                  )}
                </span>
                <div>
                  <strong>{check.label}</strong>
                  <p>{check.message}</p>
                </div>
              </div>
            ))}
          </div>
          {!report.can_analyze && !report.sample_ready && (
            <div className="upload-blockers">
              <Info size={20} />
              <div>
                <strong>
                  {report.is_sample
                    ? "Sample format checked. This is not a current household forecast."
                    : report.data_ready
                      ? "Your readings are checked. An estimate has not been generated."
                      : "A current estimate cannot be generated yet."}
                </strong>
                <ul>
                  {report.blockers.map((message) => (
                    <li key={message}>{message}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          {report.sample_ready && (
            <div className="upload-blockers">
              <Info size={20} />
              <p><strong>Sample forecast available.</strong> Select <strong>Generate sample forecast</strong> below.
                This uses synthetic data dated 1–8 January 2026. It will not estimate
                your household’s current use. The current-reading freshness check
                above does not apply to this demonstration.</p>
            </div>
          )}
          {report.can_analyze && (
            <div className="upload-ready">
              <CheckCircle2 size={21} />
              <p>
                Your readings are ready. Click{" "}
                <strong>Analyze my energy</strong> below to forecast{" "}
                {stamp(report.target_interval_start)}–
                {report.target_end?.slice(11, 16)} in {report.timezone} and plan
                your appliances.
              </p>
              <ArrowRight size={18} />
            </div>
          )}
          {!!report.warnings.length && (
            <details className="evidence-details">
              <summary>Data-quality notes ({report.warnings.length})</summary>
              {report.warnings.map((w) => (
                <p key={w}>{w}</p>
              ))}
            </details>
          )}
          <details className="evidence-details">
            <summary>Preview the uploaded readings</summary>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    {schema.columns.map((c) => (
                      <th key={c.name}>{c.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {report.preview.map((r, i) => (
                    <tr key={i}>
                      {schema.columns.map((c) => (
                        <td key={c.name}>{r[c.name] ?? "Missing"}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="microcopy">
              First five rows, ordered by timestamp. Missing and invalid cells
              are shown as missing.
            </p>
          </details>
          <button
            className="text-button"
            disabled={disabled || uploading}
            onClick={() => {
              onReport(null);
              setFile(null);
              if (input.current) input.current.value = "";
            }}
          >
            <X size={14} />
            Clear selection
          </button>
        </div>
      )}
      <details className="upload-format">
        <summary>What should my CSV contain?</summary>
        <p>
          Use one row for each minute, with ISO local timestamps such as{" "}
          <code>2026-10-31 10:15:00</code>. Choose the timezone that matches
          those timestamps. Keep the template’s column names and units.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Column</th>
                <th>Unit / format</th>
              </tr>
            </thead>
            <tbody>
              {schema.columns.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>{c.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          <Clock3 size={14} />
          Supply at least eight days and no more than 60 days. The latest
          complete interval must end within the last 30 minutes. Incomplete
          final intervals are excluded.
        </p>
        <p>
          Leave missing measurements blank. Do not replace them with made-up
          readings. A full-day consumption or bill forecast is not provided;
          this estimate covers one 15-minute interval, one hour after the latest
          usable interval.
        </p>
      </details>
    </section>
  );
}
