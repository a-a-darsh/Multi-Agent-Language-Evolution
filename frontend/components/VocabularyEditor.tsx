"use client";

import { useState } from "react";
import type { NewConcept, SemanticConcept } from "@/lib/types";

const NEW = "__new__";
const POS = ["noun", "verb", "adjective", "adverb", "interjection"];

interface Row {
  key: number;
  conceptId: string; // a known concept id, or NEW for a meaning the user invents
  word: string;
  newGloss: string;
  newPos: string;
}

interface Props {
  concepts: SemanticConcept[];
  value: Record<string, string> | undefined;
  newConcepts: NewConcept[];
  onChange: (words: Record<string, string> | undefined) => void;
  onNewConcepts: (concepts: NewConcept[]) => void;
  issuesByConcept: Record<string, string[]>;
}

let nextKey = 0;
const emptyRow = (): Row => ({ key: nextKey++, conceptId: "", word: "", newGloss: "", newPos: "noun" });

export function VocabularyEditor({ concepts, value, newConcepts, onChange, onNewConcepts, issuesByConcept }: Props) {
  // Rows are local (they include half-filled entries); seed them from the saved answers.
  const [rows, setRows] = useState<Row[]>(() => {
    const saved = [
      ...Object.entries(value ?? {}).map(([conceptId, word]) => ({ ...emptyRow(), conceptId, word })),
      ...newConcepts.map((c) => ({ ...emptyRow(), conceptId: NEW, word: c.word, newGloss: c.gloss, newPos: c.part_of_speech })),
    ];
    return saved.length ? saved : [emptyRow()];
  });
  const sorted = [...concepts].sort((a, b) => a.gloss.localeCompare(b.gloss));

  function update(next: Row[]) {
    setRows(next);
    // Only complete rows count. The backend catches conflicts with single-word questions.
    const complete = next.filter((r) => r.word.trim() && (r.conceptId === NEW ? r.newGloss.trim() : r.conceptId));
    const words = Object.fromEntries(complete.filter((r) => r.conceptId !== NEW).map((r) => [r.conceptId, r.word]));
    onChange(Object.keys(words).length ? words : undefined);
    onNewConcepts(complete.filter((r) => r.conceptId === NEW)
      .map((r) => ({ gloss: r.newGloss.trim(), part_of_speech: r.newPos, word: r.word })));
  }

  function patch(key: number, change: Partial<Row>) {
    update(rows.map((r) => (r.key === key ? { ...r, ...change } : r)));
  }

  return (
    <div className="vocab">
      {rows.map((row) => (
        <div key={row.key}>
          <div className="vocab-row">
            <select value={row.conceptId} onChange={(e) => patch(row.key, { conceptId: e.target.value })}
                    aria-label="Meaning">
              <option value="">Choose a meaning...</option>
              <option value={NEW}>+ New meaning...</option>
              {sorted.map((c) => (
                <option key={c.id} value={c.id}>{c.gloss}</option>
              ))}
            </select>
            <input type="text" placeholder="word" value={row.word} maxLength={40}
                   onChange={(e) => patch(row.key, { word: e.target.value })} aria-label="Word" />
            <button type="button" className="icon-button" aria-label="Remove word"
                    onClick={() => update(rows.length > 1 ? rows.filter((r) => r.key !== row.key) : [emptyRow()])}>
              ×
            </button>
          </div>
          {row.conceptId === NEW && (
            <div className="vocab-row new-meaning">
              <input type="text" placeholder="What does it mean? e.g. moonlight festival" value={row.newGloss}
                     maxLength={60} onChange={(e) => patch(row.key, { newGloss: e.target.value })}
                     aria-label="New meaning" autoFocus />
              <select value={row.newPos} onChange={(e) => patch(row.key, { newPos: e.target.value })}
                      aria-label="Kind of word">
                {POS.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
          )}
          {issuesByConcept[row.conceptId]?.map((message) => (
            <p key={message} className="field-error">{message}</p>
          ))}
        </div>
      ))}
      <button type="button" className="link-button" onClick={() => update([...rows, emptyRow()])}>
        + Add another word
      </button>
    </div>
  );
}
