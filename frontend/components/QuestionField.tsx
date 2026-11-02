"use client";

import type { Answer, Question, SemanticConcept } from "@/lib/types";
import { VocabularyEditor } from "./VocabularyEditor";

interface Props {
  question: Question;
  value: Answer;
  onChange: (value: Answer) => void;
  concepts: SemanticConcept[];
  errors: string[];
  issuesByConcept: Record<string, string[]>;
}

export function QuestionField({ question, value, onChange, concepts, errors, issuesByConcept }: Props) {
  const name = `q-${question.id}`;

  return (
    <fieldset className="question">
      <legend>{question.prompt}</legend>
      {question.help && <p className="help">{question.help}</p>}

      {question.input_type === "single_select" && (
        <div className="options">
          {question.options.map((o) => (
            <label key={o.value} className={`option ${value === o.value ? "selected" : ""}`}>
              <input type="radio" name={name} checked={value === o.value} onChange={() => onChange(o.value)} />
              <span className="option-label">{o.label}</span>
              {o.description && <span className="option-description">{o.description}</span>}
            </label>
          ))}
          <AiDecides checked={value === undefined} name={name} onSelect={() => onChange(undefined)} />
        </div>
      )}

      {question.input_type === "multi_select" && (
        <div className="chips">
          {question.options.map((o) => {
            const selected = Array.isArray(value) && value.includes(o.value);
            return (
              <label key={o.value} className={`chip ${selected ? "selected" : ""}`}>
                <input
                  type="checkbox"
                  checked={selected}
                  onChange={() => {
                    const current = Array.isArray(value) ? value : [];
                    const next = selected ? current.filter((v) => v !== o.value) : [...current, o.value];
                    onChange(next.length ? next : undefined);
                  }}
                />
                {o.label}
              </label>
            );
          })}
          <p className="hint">Leave all unchecked to let the AI decide.</p>
        </div>
      )}

      {question.input_type === "yes_no" && (
        <div className="options inline">
          {[
            ["Yes", true],
            ["No", false],
          ].map(([label, v]) => (
            <label key={String(label)} className={`option ${value === v ? "selected" : ""}`}>
              <input type="radio" name={name} checked={value === v} onChange={() => onChange(v as boolean)} />
              <span className="option-label">{label}</span>
            </label>
          ))}
          <AiDecides checked={value === undefined} name={name} onSelect={() => onChange(undefined)} />
        </div>
      )}

      {question.input_type === "free_text" && (
        <textarea
          rows={3}
          value={typeof value === "string" ? value : ""}
          placeholder="Leave blank to let the AI decide"
          onChange={(e) => onChange(e.target.value || undefined)}
        />
      )}

      {question.input_type === "custom_word" && (
        <>
          <input
            type="text"
            className="word-input"
            value={typeof value === "string" ? value : ""}
            placeholder="Leave blank to let the AI decide"
            onChange={(e) => onChange(e.target.value || undefined)}
          />
          {issuesByConcept[question.maps_to.split(".").pop() ?? ""]?.map((m) => (
            <p key={m} className="field-error">
              {m}
            </p>
          ))}
        </>
      )}

      {question.input_type === "custom_vocabulary" && (
        <VocabularyEditor
          concepts={concepts}
          value={value && typeof value === "object" && !Array.isArray(value) ? value : undefined}
          issuesByConcept={issuesByConcept}
          onChange={(words) => onChange(words)}
        />
      )}

      {errors.map((m) => (
        <p key={m} className="field-error">
          {m}
        </p>
      ))}
    </fieldset>
  );
}

function AiDecides({ checked, name, onSelect }: { checked: boolean; name: string; onSelect: () => void }) {
  return (
    <label className={`option ai ${checked ? "selected" : ""}`}>
      <input type="radio" name={name} checked={checked} onChange={onSelect} />
      <span className="option-label">Let the AI decide</span>
    </label>
  );
}
