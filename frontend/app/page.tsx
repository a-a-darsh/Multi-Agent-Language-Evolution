"use client";

import { useEffect, useState } from "react";
import { LlmSetup } from "@/components/LlmSetup";
import { QuestionField } from "@/components/QuestionField";
import { WorldView } from "@/components/WorldView";
import { ApiError, generateWorld, getConcepts, getLlmSettings, getQuestionnaire } from "@/lib/api";
import type {
  Answer,
  Answers,
  LLMSettingsView,
  Questionnaire,
  SemanticConcept,
  ValidationIssue,
  World,
} from "@/lib/types";

const PROVIDER_NAMES = { local: "Ollama", openai: "OpenAI", anthropic: "Anthropic", openrouter: "OpenRouter" };

export default function Home() {
  const [llm, setLlm] = useState<LLMSettingsView | null>(null);
  const [editingLlm, setEditingLlm] = useState(false);
  const [questionnaire, setQuestionnaire] = useState<Questionnaire | null>(null);
  const [concepts, setConcepts] = useState<SemanticConcept[]>([]);
  const [answers, setAnswers] = useState<Answers>({});
  const [issues, setIssues] = useState<ValidationIssue[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [world, setWorld] = useState<World | null>(null);

  async function loadQuestions() {
    setError(null);
    setIssues([]);
    setAnswers({});
    setWorld(null);
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
      const result = await generateWorld(questionnaire.seed, answers);
      setWorld(result.world);
      window.scrollTo({ top: 0 });
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.message);
        setIssues(e.body.issues);
      } else {
        setError(String(e));
      }
    } finally {
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

  return (
    <main>
      <header className="page-header">
        <div className="title-row">
          <h1>Language Evolution Simulator</h1>
          {llm?.configured && !editingLlm && (
            <p className="llm-status">
              {PROVIDER_NAMES[llm.provider!]} · <code>{llm.model}</code>{" "}
              <button type="button" className="link-button" onClick={() => setEditingLlm(true)}>
                Change
              </button>
            </p>
          )}
        </div>
        <p>Describe a civilization. Skip anything you like and the AI will decide.</p>
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
      ) : world ? (
        <>
          <WorldView world={world} />
          <div className="notice">
            Language generation isn&apos;t available yet. It arrives with the morphology engine and the
            language design agent. The language will be based only on your civilization, not its neighbors.
          </div>
          <div className="actions">
            <button type="button" className="secondary" onClick={() => setWorld(null)}>
              Back to my answers
            </button>
            <button type="button" onClick={loadQuestions}>
              Start over
            </button>
          </div>
        </>
      ) : (
        <form onSubmit={submit}>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}

          {!questionnaire && !error && <p className="hint">Loading questions...</p>}

          {questionnaire?.questions.map((q) => (
            <QuestionField
              key={`${questionnaire.seed}-${q.id}`}
              question={q}
              value={answers[q.id]}
              onChange={setAnswer(q.id)}
              concepts={concepts}
              errors={byObject(q.id)}
              issuesByConcept={issuesByConcept}
            />
          ))}

          {questionnaire && (
            <div className="actions">
              <button type="button" className="secondary" onClick={loadQuestions} disabled={submitting}>
                Different questions
              </button>
              <button type="submit" disabled={submitting}>
                {submitting ? "Generating..." : "Generate civilization"}
              </button>
            </div>
          )}
        </form>
      )}
    </main>
  );
}
