import { humanize } from "@/lib/format";
import type { LanguageOverview, Sound, ValidationIssue } from "@/lib/types";

const POSITIONS: Record<string, string> = {
  onset: "start a syllable",
  coda: "end a syllable",
  word_initial: "start a word",
  word_final: "end a word",
};

export function OverviewTab({ language, onOpenDictionary }: { language: LanguageOverview; onOpenDictionary: () => void }) {
  const l = language;
  return (
    <div className="overview">
      {l.summary && <p className="lead">{l.summary}</p>}

      <section className="card">
        <h3>Sounds</h3>
        <SoundRow label="Consonants" sounds={l.consonants} />
        <SoundRow label="Vowels" sounds={l.vowels} />
        <p className="hint">Each letter is always pronounced the same way. Highlighted sounds are especially common.</p>
        <dl className="facts">
          <dt>Syllables</dt>
          <dd>
            {l.syllable_patterns.join(", ")}
            <span className="hint"> (C = consonant, V = vowel)</span>
          </dd>
          <dt>Vowels together</dt>
          <dd>{l.allow_hiatus ? "Allowed (as in ri·a)" : "Not allowed"}</dd>
          {l.restrictions.map((r) => (
            <FactRow key={r.letter} label={`“${r.letter}”`}>
              never {r.not_allowed.map((p) => POSITIONS[p] ?? p).join(" or ")}
            </FactRow>
          ))}
        </dl>
      </section>

      <section className="card">
        <h3>Grammar</h3>
        <dl className="facts">
          <dt>Word order</dt>
          <dd>{l.word_order}</dd>
          <dt>Word building</dt>
          <dd>{humanize(l.morphological_type)}</dd>
          {l.adposition_type && (
            <>
              <dt>Adpositions</dt>
              <dd>{humanize(l.adposition_type)}</dd>
            </>
          )}
        </dl>
        {l.features.length > 0 && (
          <ul className="feature-list">
            {l.features.map((f) => (
              <li key={f.name}>
                <strong>{humanize(f.name)}</strong> — {f.description}{" "}
                <span className="hint">({f.realization})</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card">
        <div className="section-head">
          <h3>Word-building affixes</h3>
          <button type="button" className="link-button" onClick={onOpenDictionary}>
            Browse the dictionary →
          </button>
        </div>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Affix</th>
                <th>Meaning</th>
                <th>Makes</th>
                <th>Examples</th>
              </tr>
            </thead>
            <tbody>
              {l.affixes.map((a) => (
                <tr key={a.rule_id}>
                  <td><code className="word">{a.form}</code></td>
                  <td>{a.meaning}</td>
                  <td className="nowrap">{a.takes.join("/")} → {a.makes}</td>
                  <td>
                    {a.examples.length
                      ? a.examples.map((e) => (
                          <span key={e.id} className="example">
                            <span className="word">{e.word}</span> {e.gloss}
                          </span>
                        ))
                      : <span className="hint">none yet</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {(l.linking_vowel || l.linking_consonant) && (
          <p className="hint">
            When pieces can&apos;t join directly, the language inserts
            {l.linking_vowel && <> the vowel <code>{l.linking_vowel}</code></>}
            {l.linking_vowel && l.linking_consonant && " or"}
            {l.linking_consonant && <> the consonant <code>{l.linking_consonant}</code></>}.
          </p>
        )}
      </section>

      {l.warnings.length > 0 && (
        <details className="card warnings">
          <summary>{l.warnings.length} note{l.warnings.length > 1 ? "s" : ""} from generation</summary>
          {/* Grouped by kind: one run produced 19 identical-kind notes that buried the rest. */}
          {groupByCode(l.warnings).map(([code, items]) => (
            <details key={code} className="note-group">
              <summary>
                {NOTE_LABELS[code] ?? humanize(code.toLowerCase())} <span className="hint">×{items.length}</span>
              </summary>
              <ul className="issue-list">
                {items.map((w, i) => <li key={i}>{w.message}</li>)}
              </ul>
            </details>
          ))}
        </details>
      )}
    </div>
  );
}

const NOTE_LABELS: Record<string, string> = {
  RULE_CORRECTED: "Affix choice corrected automatically",
  IMPLAUSIBLE_DERIVATION: "Derivation rejected by the reviewer (given its own word)",
  ILLEGAL_DERIVED_FORM: "Derivation impossible with this sound system (given its own word)",
  CRITIC_INCOHERENT_GRAMMAR: "Reviewer's opinion on the grammar",
  CRITIC_INFLECTIONAL_AFFIX: "Affix looks grammatical rather than word-building",
  ADJUSTED_FOR_USER_WORDS: "Sounds added so your words stay valid",
  USER_WORD_KEPT: "Your word kept instead of a derivation",
  USER_POLYSEMY: "Your word used for several meanings",
  DERIVATION_TARGET_MISSED: "Fewer derived words than requested",
  BASE_IS_SHARED_WORD: "Derivation rejected: built on a shared word (given its own word)",
  DERIVATION_KEPT: "Derived word kept instead of a shared word",
  SENSE_POS_MISMATCH: "Shared word rejected: different kinds of word (given its own word)",
  EXTENSION_CHAIN: "Shared word rejected: it borrowed a borrowed word (given its own word)",
};

function groupByCode(warnings: ValidationIssue[]): [string, ValidationIssue[]][] {
  const groups = new Map<string, ValidationIssue[]>();
  for (const w of warnings) groups.set(w.code, [...(groups.get(w.code) ?? []), w]);
  return [...groups.entries()];
}

function SoundRow({ label, sounds }: { label: string; sounds: Sound[] }) {
  return (
    <div className="sound-row">
      <span className="sound-label">{label}</span>
      <div className="sounds-grid">
        {sounds.map((s) => (
          <span key={s.letter} className={s.common ? "sound common" : "sound"} title={`/${s.ipa}/`}>
            {s.letter}
            {s.ipa !== s.letter && <small>/{s.ipa}/</small>}
          </span>
        ))}
      </div>
    </div>
  );
}

function FactRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </>
  );
}
