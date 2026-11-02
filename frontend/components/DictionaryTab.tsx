"use client";

import { useEffect, useState } from "react";
import { ApiError, getEntry, searchDictionary } from "@/lib/api";
import { humanize } from "@/lib/format";
import type { EntryDetail, EntryPage, EntrySummary } from "@/lib/types";

const ORIGIN_LABEL: Record<string, string> = {
  generated_root: "root",
  derived: "derived",
  user_defined: "your word",
};

const POS = ["noun", "verb", "adjective", "adverb", "pronoun", "numeral", "adposition", "particle", "interjection"];

export function DictionaryTab({ languageId }: { languageId: string }) {
  const [q, setQ] = useState("");
  const [pos, setPos] = useState("");
  const [origin, setOrigin] = useState("");
  const [page, setPage] = useState<EntryPage | null>(null);
  const [selected, setSelected] = useState<EntryDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Small debounce so typing doesn't fire a request per keystroke.
    const timer = setTimeout(() => {
      searchDictionary(languageId, { q, pos, origin, page_size: 200 })
        .then(setPage)
        .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
    }, 200);
    return () => clearTimeout(timer);
  }, [languageId, q, pos, origin]);

  function open(entry: EntrySummary) {
    getEntry(languageId, entry.id)
      .then(setSelected)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }

  return (
    <div className="dictionary">
      <div className="filters">
        <input type="text" placeholder="Search a word or its meaning..." value={q} onChange={(e) => setQ(e.target.value)}
               aria-label="Search" />
        <select value={pos} onChange={(e) => setPos(e.target.value)} aria-label="Part of speech">
          <option value="">All parts of speech</option>
          {POS.map((p) => <option key={p} value={p}>{humanize(p)}</option>)}
        </select>
        <select value={origin} onChange={(e) => setOrigin(e.target.value)} aria-label="Origin">
          <option value="">All words</option>
          <option value="derived">Derived words</option>
          <option value="generated_root">Roots</option>
          <option value="user_defined">Your words</option>
        </select>
      </div>
      {error && <div className="error-banner">{error}</div>}

      <div className="dictionary-body">
        <div className="entry-list">
          <p className="hint">{page ? `${page.total} entries` : "Loading..."}</p>
          <ul>
            {page?.entries.map((e) => (
              <li key={e.id}>
                <button type="button" className={selected?.id === e.id ? "entry-row selected" : "entry-row"}
                        onClick={() => open(e)}>
                  <span className="word">{e.word}</span>
                  <span className="gloss">{e.gloss}</span>
                  <span className={`origin ${e.origin}`}>{ORIGIN_LABEL[e.origin]}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        <aside className="entry-detail">
          {selected ? <EntryCard entry={selected} onOpen={open} /> : (
            <p className="hint">Select a word to see how it is built.</p>
          )}
        </aside>
      </div>
    </div>
  );
}

function EntryCard({ entry, onOpen }: { entry: EntryDetail; onOpen: (e: EntrySummary) => void }) {
  const link = (e: EntrySummary) => (
    <button type="button" className="word-link" onClick={() => onOpen(e)}>
      <span className="word">{e.word}</span> <span className="gloss">{e.gloss}</span>
    </button>
  );
  const isLong = entry.segments.length > 1;

  return (
    <section className="card">
      <h2 className="word big">{entry.word}</h2>
      <p className="hint">
        {entry.syllables.join("·")} · {entry.part_of_speech} · {ORIGIN_LABEL[entry.origin]}
      </p>
      <p className="meaning">{entry.gloss}</p>
      {entry.concept_ids.length > 1 && (
        <p className="hint">One word with {entry.concept_ids.length} related meanings.</p>
      )}

      {isLong && (
        <>
          <h4>How it&apos;s built</h4>
          <div className="segments">
            {entry.segments.map((s, i) => (
              <span key={i} className={`segment ${s.kind}`}>
                <span className="word">{s.kind === "prefix" ? `${s.text}-` : s.kind === "suffix" ? `-${s.text}` : s.text}</span>
                <small>{s.kind === "linking" ? "linking sound" : s.gloss}</small>
              </span>
            ))}
          </div>
        </>
      )}

      {entry.derived_from && entry.rule && (
        <p>
          Built from {link(entry.derived_from)} with the <strong>{entry.rule.name.toLowerCase()}</strong> affix
          ({entry.rule.meaning}).
        </p>
      )}

      {entry.derived_words.length > 0 && (
        <>
          <h4>Words built from it</h4>
          <ul className="related">{entry.derived_words.map((e) => <li key={e.id}>{link(e)}</li>)}</ul>
        </>
      )}

      {entry.family.length > 1 && (
        <>
          <h4>Word family of {entry.family_root.word}</h4>
          <ul className="related">
            {entry.family.map((e) => (
              <li key={e.id} className={e.id === entry.id ? "current" : undefined}>{link(e)}</li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
