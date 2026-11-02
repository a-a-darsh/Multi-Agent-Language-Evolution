"use client";

import { useState } from "react";
import type { SemanticConcept } from "@/lib/types";

interface Row {
  key: number;
  conceptId: string;
  word: string;
}

interface Props {
  concepts: SemanticConcept[];
  value: Record<string, string> | undefined;
  onChange: (words: Record<string, string> | undefined) => void;
  issuesByConcept: Record<string, string[]>;
}

let nextKey = 0;
const emptyRow = (): Row => ({ key: nextKey++, conceptId: "", word: "" });

export function VocabularyEditor({ concepts, value, onChange, issuesByConcept }: Props) {
  // Rows are local (they include half-filled entries); seed them from the saved answer.
  const [rows, setRows] = useState<Row[]>(() => {
    const saved = Object.entries(value ?? {}).map(([conceptId, word]) => ({ key: nextKey++, conceptId, word }));
    return saved.length ? saved : [emptyRow()];
  });
  const sorted = [...concepts].sort((a, b) => a.gloss.localeCompare(b.gloss));

  function update(next: Row[]) {
    setRows(next);
    // Only complete rows count. Duplicate meanings keep the last one; the
    // backend catches conflicts with single-word questions.
    const words = Object.fromEntries(
      next.filter((r) => r.conceptId && r.word.trim()).map((r) => [r.conceptId, r.word]),
    );
    onChange(Object.keys(words).length ? words : undefined);
  }

  function patch(key: number, change: Partial<Row>) {
    update(rows.map((r) => (r.key === key ? { ...r, ...change } : r)));
  }

  return (
    <div className="vocab">
      {rows.map((row) => (
        <div key={row.key}>
          <div className="vocab-row">
            <select
              value={row.conceptId}
              onChange={(e) => patch(row.key, { conceptId: e.target.value })}
              aria-label="Meaning"
            >
              <option value="">Choose a meaning...</option>
              {sorted.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.gloss}
                </option>
              ))}
            </select>
            <input
              type="text"
              placeholder="word"
              value={row.word}
              onChange={(e) => patch(row.key, { word: e.target.value })}
              aria-label="Word"
            />
            <button
              type="button"
              className="icon-button"
              onClick={() => update(rows.length > 1 ? rows.filter((r) => r.key !== row.key) : [emptyRow()])}
              aria-label="Remove word"
            >
              ×
            </button>
          </div>
          {issuesByConcept[row.conceptId]?.map((message) => (
            <p key={message} className="field-error">
              {message}
            </p>
          ))}
        </div>
      ))}
      <button type="button" className="link-button" onClick={() => update([...rows, emptyRow()])}>
        + Add another word
      </button>
    </div>
  );
}
