"use client";

import type { Answer, NewConcept, Question, SemanticConcept } from "@/lib/types";
import { VocabularyEditor } from "./VocabularyEditor";

interface Props {
  question: Question;
  value: Answer;
  onChange: (value: Answer) => void;
  // "Describe your own" text for multiple-choice questions; undefined = not in use.
  customValue: string | undefined;
  onCustomChange: (value: string | undefined) => void;
  newConcepts: NewConcept[];
  onNewConcepts: (concepts: NewConcept[]) => void;
  concepts: SemanticConcept[];
  errors: string[];
  issuesByConcept: Record<string, string[]>;
}

export function QuestionField(props: Props) {
  const { question, value, onChange, customValue, onCustomChange, concepts, errors, issuesByConcept } = props;
  const name = `q-${question.id}`;
  const customOn = customValue !== undefined;

  const customInput = customOn && (
    <input
      type="text"
      className="custom-input"
      autoFocus
      maxLength={300}
      value={customValue}
      placeholder="Describe it in your own words"
      onChange={(e) => onCustomChange(e.target.value)}
    />
  );

  return (
    <fieldset className="question">
      <legend>{question.prompt}</legend>
      {question.help && <p className="help">{question.help}</p>}

      {question.input_type === "single_select" && (
        <>
          <div className="options">
            {question.options.map((o) => (
              <label key={o.value} className={`option ${value === o.value ? "selected" : ""}`}>
                <input type="radio" name={name} checked={value === o.value}
                       onChange={() => { onChange(o.value); onCustomChange(undefined); }} />
                <span className="option-label">{o.label}</span>
                {o.description && <span className="option-description">{o.description}</span>}
              </label>
            ))}
            {question.allow_custom && (
              <label className={`option custom ${customOn ? "selected" : ""}`}>
                <input type="radio" name={name} checked={customOn}
                       onChange={() => { onChange(undefined); onCustomChange(""); }} />
                <span className="option-label">Describe your own</span>
              </label>
            )}
            <AiDecides checked={value === undefined && !customOn} name={name}
                       onSelect={() => { onChange(undefined); onCustomChange(undefined); }} />
          </div>
          {customInput}
        </>
      )}

      {question.input_type === "multi_select" && (
        <>
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
            {question.allow_custom && (
              <label className={`chip custom ${customOn ? "selected" : ""}`}>
                <input type="checkbox" checked={customOn}
                       onChange={() => onCustomChange(customOn ? undefined : "")} />
                Something else...
              </label>
            )}
            <p className="hint">Leave everything unchecked to let the AI decide.</p>
          </div>
          {customInput}
        </>
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

      {(question.input_type === "short_text" || question.input_type === "custom_word") && (
        <>
          <input
            type="text"
            className="word-input"
            maxLength={question.input_type === "short_text" ? 60 : 40}
            value={typeof value === "string" ? value : ""}
            placeholder="Leave blank to let the AI decide"
            onChange={(e) => onChange(e.target.value || undefined)}
          />
          {question.input_type === "custom_word" &&
            issuesByConcept[question.maps_to.split(".").pop() ?? ""]?.map((m) => (
              <p key={m} className="field-error">{m}</p>
            ))}
        </>
      )}

      {question.input_type === "custom_vocabulary" && (
        <VocabularyEditor
          concepts={concepts}
          value={value && typeof value === "object" && !Array.isArray(value) ? value : undefined}
          newConcepts={props.newConcepts}
          issuesByConcept={issuesByConcept}
          onChange={(words) => onChange(words)}
          onNewConcepts={props.onNewConcepts}
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
