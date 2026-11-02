"use client";

import { useEffect, useState } from "react";
import { ApiError, getGenerationLog } from "@/lib/api";
import type { GenerationLog } from "@/lib/types";

export function GenerationLogTab({ languageId }: { languageId: string }) {
  const [log, setLog] = useState<GenerationLog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getGenerationLog(languageId)
      .then(setLog)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [languageId]);

  if (error) return <div className="error-banner">{error}</div>;
  if (!log) return <p className="hint">Loading...</p>;

  const seconds = log.llm_calls.reduce((t, c) => t + c.seconds, 0);
  return (
    <div className="overview">
      <p className="hint">
        Seed {log.seed} · {log.llm_calls.length} model calls · {Math.round(seconds)}s of model time.
        Each step below is a node in the agent workflow; problems found by checks were sent back for revision.
      </p>
      <section className="card">
        <h3>Steps</h3>
        <ol className="log-steps">
          {log.steps.map((s, i) => (
            <li key={i}>
              <span className="badge neutral">{s.agent}</span> <strong>{s.step}</strong>{" "}
              {s.issues.length === 0 ? <span className="hint">ok</span> : (
                <ul className="issue-list">
                  {s.issues.map((issue, j) => (
                    <li key={j}><code>{issue.code}</code> {issue.message}</li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ol>
      </section>
      <section className="card">
        <h3>Model calls</h3>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr><th>Agent</th><th>Step</th><th>Model</th><th>Time</th><th>Result</th></tr>
            </thead>
            <tbody>
              {log.llm_calls.map((c, i) => (
                <tr key={i}>
                  <td>{c.agent}</td>
                  <td>{c.step}</td>
                  <td className="model-name">{c.model}</td>
                  <td className="nowrap">{c.seconds}s</td>
                  <td>{c.error ? <span className="field-error">{c.error.slice(0, 120)}</span> : "ok"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
