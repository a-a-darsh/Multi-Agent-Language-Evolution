"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { DictionaryTab } from "@/components/DictionaryTab";
import { GenerationLogTab } from "@/components/GenerationLogTab";
import { OverviewTab } from "@/components/OverviewTab";
import { WorldView } from "@/components/WorldView";
import { ApiError, getLanguage, getLanguageWorld } from "@/lib/api";
import { formatYear, humanize } from "@/lib/format";
import type { LanguageOverview, World } from "@/lib/types";

const TABS = ["Overview", "Dictionary", "World", "Generation log"] as const;
type Tab = (typeof TABS)[number];

export default function LanguagePage() {
  const { id } = useParams<{ id: string }>();
  const [language, setLanguage] = useState<LanguageOverview | null>(null);
  const [world, setWorld] = useState<World | null>(null);
  const [tab, setTab] = useState<Tab>("Overview");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getLanguage(id), getLanguageWorld(id)])
      .then(([l, w]) => {
        setLanguage(l);
        setWorld(w);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [id]);

  if (error) return <main><div className="error-banner">{error}</div></main>;
  if (!language) return <main><p className="hint">Loading...</p></main>;

  const total = Object.values(language.counts).reduce((a, b) => a + b, 0);

  return (
    <main className="wide">
      <header className="page-header">
        <p className="eyebrow">
          Language of the {language.civilization.name}
          {language.civilization.era &&
            ` · ${humanize(language.civilization.era)}, ${formatYear(language.civilization.founding_year)}`}
        </p>
        <h1>{language.name}</h1>
        <p>
          {total} words · {language.counts.compound ?? 0} compounds · {language.counts.derived} derived ·{" "}
          {language.counts.user_defined} of your own
        </p>
      </header>

      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "tab active" : "tab"}
                  onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {tab === "Overview" && <OverviewTab language={language} onOpenDictionary={() => setTab("Dictionary")} />}
      {tab === "Dictionary" && <DictionaryTab languageId={id} />}
      {tab === "World" && world && <WorldView world={world} />}
      {tab === "Generation log" && <GenerationLogTab languageId={id} />}
    </main>
  );
}
