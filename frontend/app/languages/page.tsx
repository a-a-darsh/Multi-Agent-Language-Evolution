"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ApiError, listLanguages } from "@/lib/api";
import type { LanguageSummary } from "@/lib/types";

export default function LanguagesPage() {
  const [languages, setLanguages] = useState<LanguageSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listLanguages()
      .then(setLanguages)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  return (
    <main>
      <header className="page-header">
        <h1>My languages</h1>
      </header>
      {error && <div className="error-banner">{error}</div>}
      {languages?.length === 0 && (
        <p className="hint">
          No languages yet. <Link href="/">Create one</Link>.
        </p>
      )}
      <div className="language-list">
        {languages?.map((l) => (
          <Link key={l.id} href={`/languages/${l.id}`} className="card language-card">
            <h2>{l.name}</h2>
            <p className="hint">
              Spoken by the {l.civilization_name} · {l.entry_count} words, {l.derived_count} derived ·{" "}
              {new Date(l.created_at).toLocaleDateString()}
            </p>
          </Link>
        ))}
      </div>
    </main>
  );
}
