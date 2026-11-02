"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ApiError, getGeneration } from "@/lib/api";
import type { GenerationSummary } from "@/lib/types";

// Graph node names (reported by LangGraph's stream) -> what the user sees.
const STAGES: { label: string; nodes: string[] }[] = [
  { label: "Creating the civilization and its neighbors", nodes: ["world"] },
  { label: "Reading your answers", nodes: ["design:analyze_inputs"] },
  { label: "Designing sounds and grammar", nodes: ["design:design", "design:repair"] },
  { label: "Checking and reviewing the design", nodes: ["design:validate", "design:critique", "design:finalize"] },
  { label: "Choosing vocabulary", nodes: ["dictionary:select_concepts"] },
  { label: "Planning word families", nodes: ["dictionary:plan", "dictionary:repair", "dictionary:expand"] },
  { label: "Building and reviewing words", nodes: ["dictionary:build", "dictionary:critique", "dictionary:finalize"] },
];

const POLL_MS = 1500;

export default function GenerationProgress() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [job, setJob] = useState<GenerationSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startedAt] = useState(() => Date.now());
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await getGeneration(id);
        if (cancelled) return;
        setJob(next);
        if (next.status === "succeeded" && next.language_id) {
          router.replace(`/languages/${next.language_id}`);
          return;
        }
        if (next.status !== "failed") timer = setTimeout(poll, POLL_MS);
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError ? e.message : String(e));
      }
    }
    poll();
    const clock = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      clearInterval(clock);
    };
  }, [id, router]);

  const steps = job?.steps ?? [];
  const reached = STAGES.map((s) => s.nodes.some((n) => steps.includes(n)));
  const current = reached.lastIndexOf(true);
  const repairs = steps.filter((s) => s.endsWith(":repair")).length;
  const elapsed = Math.round((now - startedAt) / 1000);

  return (
    <main>
      <header className="page-header">
        <h1>Generating your language</h1>
        <p>A local model usually takes 2 to 5 minutes. You can leave this page; the job keeps running.</p>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      {job?.status === "failed" ? (
        <section className="card">
          <h2>Generation failed</h2>
          <ul className="issue-list">
            {job.errors.map((e, i) => (
              <li key={i}>
                <code>{e.code}</code> {e.message}
              </li>
            ))}
          </ul>
          <div className="actions">
            <Link href="/" className="button">
              Try again
            </Link>
          </div>
        </section>
      ) : (
        <section className="card">
          <ol className="progress">
            {STAGES.map((stage, i) => {
              const state = i < current ? "done" : i === current ? "active" : "pending";
              return (
                <li key={stage.label} className={state}>
                  <span className="dot" aria-hidden />
                  {stage.label}
                </li>
              );
            })}
          </ol>
          <p className="hint">
            {elapsed}s elapsed
            {repairs > 0 && ` · ${repairs} revision${repairs > 1 ? "s" : ""} after checks found problems`}
          </p>
        </section>
      )}
    </main>
  );
}
