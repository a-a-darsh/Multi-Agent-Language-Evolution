"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { LlmSetup } from "@/components/LlmSetup";
import { QuestionField } from "@/components/QuestionField";
import { ApiError, getConcepts, getLlmSettings, getQuestionnaire, startGeneration } from "@/lib/api";
import type {
  Answer,
  Answers,
  LLMSettingsView,
  NewConcept,
  Questionnaire,
  SemanticConcept,
  ValidationIssue,
} from "@/lib/types";

const PROVIDER_NAMES = { local: "Ollama", openai: "OpenAI", anthropic: "Anthropic", openrouter: "OpenRouter" };

export default function CreatePage() {
  const router = useRouter();
  const [llm, setLlm] = useState<LLMSettingsView | null>(null);
  const [editingLlm, setEditingLlm] = useState(false);
  const [questionnaire, setQuestionnaire] = useState<Questionnaire | null>(null);
  const [concepts, setConcepts] = useState<SemanticConcept[]>([]);
  const [answers, setAnswers] = useState<Answers>({});
  const [customAnswers, setCustomAnswers] = useState<Record<string, string | undefined>>({});
  const [newConcepts, setNewConcepts] = useState<NewConcept[]>([]);
  const [derivedPercent, setDerivedPercent] = useState(20);
  const [issues, setIssues] = useState<ValidationIssue[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function loadQuestions() {
    setError(null);
    setIssues([]);
    setAnswers({});
    setCustomAnswers({});
    setNewConcepts([]);
    try {
      const [q, c] = await Promise.all([getQuestionnaire(), concepts.length ? concepts : getConcepts()]);
      setQuestionnaire(q);
      setConcepts(c);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }

  useEffect(() => {
    getLlmSettings()
      .then(setLlm)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    loadQuestions();
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!questionnaire) return;
    setSubmitting(true);
    setError(null);
    setIssues([]);
    try {
      const written = Object.fromEntries(
        Object.entries(customAnswers).filter((e): e is [string, string] => !!e[1]?.trim()),
      );
      const { generation_id } = await startGeneration(questionnaire.seed, answers, written, newConcepts,
                                                      derivedPercent / 100);
      router.push(`/generations/${generation_id}`);
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.message);
        setIssues(e.body.issues);
      } else {
        setError(String(e));
      }
      setSubmitting(false);
    }
  }

  const setAnswer = (id: string) => (value: Answer) => setAnswers((prev) => ({ ...prev, [id]: value }));

  // Issues point at a question id (bad answer) or a concept id (bad custom word).
  const byObject = (id: string) => issues.filter((i) => i.object_id === id).map((i) => i.message);
  const issuesByConcept: Record<string, string[]> = {};
  for (const c of concepts) {
    const messages = byObject(c.id);
    if (messages.length) issuesByConcept[c.id] = messages;
  }
  // Anything that doesn't point at a question or a known meaning (e.g. a new meaning
  // added twice) is listed in the banner so it isn't silently lost.
  const placed = new Set([...(questionnaire?.questions.map((q) => q.id) ?? []), ...Object.keys(issuesByConcept)]);
  const unplaced = issues.filter((i) => !i.object_id || !placed.has(i.object_id));

  return (
    <main>
      <header className="page-header">
        <div className="title-row">
          <h1>Create a language</h1>
          {llm?.configured && !editingLlm && (
            <p className="llm-status">
              {PROVIDER_NAMES[llm.provider!]} · <code>{llm.model}</code>{" "}
              <button type="button" className="link-button" onClick={() => setEditingLlm(true)}>
                Change
              </button>
            </p>
          )}
        </div>
        <p>Describe a civilization and its speech. Skip anything you like and the AI will decide.</p>
      </header>

      {llm && (!llm.configured || editingLlm) ? (
        <LlmSetup
          current={llm}
          onSaved={(saved) => {
            setLlm(saved);
            setEditingLlm(false);
          }}
          onCancel={llm.configured ? () => setEditingLlm(false) : undefined}
        />
      ) : !llm ? (
        error ? (
          <div className="error-banner" role="alert">
            {error}
          </div>
        ) : (
          <p className="hint">Loading...</p>
        )
      ) : (
        <form onSubmit={submit}>
          {error && (
            <div className="error-banner" role="alert">
              {error}
              {unplaced.length > 0 && (
                <ul className="issue-list">{unplaced.map((i, n) => <li key={n}>{i.message}</li>)}</ul>
              )}
            </div>
          )}

          {!questionnaire && !error && <p className="hint">Loading questions...</p>}

          {questionnaire?.questions.map((q) => (
            <QuestionField
              key={`${questionnaire.seed}-${q.id}`}
              question={q}
              value={answers[q.id]}
              onChange={setAnswer(q.id)}
              customValue={customAnswers[q.id]}
              onCustomChange={(v) => setCustomAnswers((prev) => ({ ...prev, [q.id]: v }))}
              newConcepts={newConcepts}
              onNewConcepts={setNewConcepts}
              concepts={concepts}
              errors={byObject(q.id)}
              issuesByConcept={issuesByConcept}
            />
          ))}

          {questionnaire && (
            <fieldset className="question">
              <legend>Options</legend>
              <label className="slider-field">
                <span>
                  Words built from other words: <strong>{derivedPercent}%</strong>
                </span>
                <input type="range" min={0} max={50} step={5} value={derivedPercent}
                       onChange={(e) => setDerivedPercent(Number(e.target.value))} />
                <span className="hint">
                  {derivedPercent === 0
                    ? "Every word gets its own unrelated root."
                    : `About ${derivedPercent}% of words will be built from others (sea → sailor). ` +
                      "Derivations that don't make sense are still rejected, so the result can come in lower."}
                </span>
              </label>
            </fieldset>
          )}

          {questionnaire && (
            <div className="actions">
              <button type="button" className="secondary" onClick={loadQuestions} disabled={submitting}>
                Different questions
              </button>
              <button type="submit" disabled={submitting}>
                {submitting ? "Starting..." : "Generate language"}
              </button>
            </div>
          )}
        </form>
      )}
    </main>
  );
}
