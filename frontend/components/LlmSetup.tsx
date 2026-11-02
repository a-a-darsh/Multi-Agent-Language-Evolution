"use client";

import { useState } from "react";
import { ApiError, listLlmModels, saveLlmSettings } from "@/lib/api";
import type { LLMConnection, LLMProvider, LLMSettingsView } from "@/lib/types";

const PROVIDERS: { value: LLMProvider; label: string; description: string }[] = [
  { value: "local", label: "Local (Ollama)", description: "Runs on your computer. Free and private." },
  { value: "openai", label: "OpenAI", description: "GPT models. Needs an API key." },
  { value: "anthropic", label: "Anthropic", description: "Claude models. Needs an API key." },
  { value: "openrouter", label: "OpenRouter", description: "Many providers through one API key." },
];

const DEFAULT_LOCAL_URL = "http://localhost:11434";

interface Props {
  current: LLMSettingsView;
  onSaved: (settings: LLMSettingsView) => void;
  onCancel?: () => void;
}

export function LlmSetup({ current, onSaved, onCancel }: Props) {
  const [provider, setProvider] = useState<LLMProvider>(current.provider ?? "local");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState(current.base_url ?? DEFAULT_LOCAL_URL);
  const [models, setModels] = useState<string[] | null>(null);
  const [model, setModel] = useState(current.model ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const isLocal = provider === "local";
  // A saved key for this provider can be reused, so the field may be left blank.
  const hasSavedKey = current.provider === provider && current.api_key_hint !== null;

  function connection(): LLMConnection {
    return isLocal ? { provider, base_url: baseUrl } : { provider, api_key: apiKey || undefined };
  }

  function chooseProvider(p: LLMProvider) {
    setProvider(p);
    setModels(null);
    setModel(p === current.provider ? (current.model ?? "") : "");
    setError(null);
  }

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const connect = () =>
    run(async () => {
      const result = await listLlmModels(connection());
      setModels(result.models);
      if (!result.models.includes(model)) setModel(result.models[0] ?? "");
    });

  const save = () => run(async () => onSaved(await saveLlmSettings({ ...connection(), model })));

  return (
    <section className="card setup">
      <p className="eyebrow">Setup</p>
      <h2>Which AI model should design your languages?</h2>
      <p className="hint">You can change this later.</p>

      <div className="options">
        {PROVIDERS.map((p) => (
          <label key={p.value} className={`option ${provider === p.value ? "selected" : ""}`}>
            <input type="radio" name="provider" checked={provider === p.value} onChange={() => chooseProvider(p.value)} />
            <span className="option-label">{p.label}</span>
            <span className="option-description">{p.description}</span>
          </label>
        ))}
      </div>

      <div className="setup-fields">
        {isLocal ? (
          <label className="field">
            <span>Ollama server address</span>
            <input type="text" value={baseUrl} onChange={(e) => { setBaseUrl(e.target.value); setModels(null); }} />
          </label>
        ) : (
          <label className="field">
            <span>API key</span>
            <input
              type="password"
              autoComplete="off"
              value={apiKey}
              placeholder={hasSavedKey ? `Saved key ${current.api_key_hint} (leave blank to keep it)` : "Paste your API key"}
              onChange={(e) => { setApiKey(e.target.value); setModels(null); }}
            />
            <span className="hint">
              Stored only on this computer, in the backend&apos;s settings file. It is never shown again.
            </span>
          </label>
        )}

        <button type="button" className="secondary" onClick={connect} disabled={busy || (!isLocal && !apiKey && !hasSavedKey)}>
          {busy && !models ? "Connecting..." : models ? "Refresh models" : "Connect"}
        </button>

        {models && (
          <label className="field">
            <span>Model ({models.length} available)</span>
            <input type="text" list="llm-models" value={model} onChange={(e) => setModel(e.target.value)} />
            <datalist id="llm-models">
              {models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>
        )}
      </div>

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      <div className="actions">
        {onCancel && (
          <button type="button" className="secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
        )}
        <button type="button" onClick={save} disabled={busy || !models || !models.includes(model)}>
          Save and continue
        </button>
      </div>
    </section>
  );
}
