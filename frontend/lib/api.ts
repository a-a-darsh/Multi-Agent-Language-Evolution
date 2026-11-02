import type {
  Answers,
  ApiErrorBody,
  EntryDetail,
  EntryPage,
  GenerationLog,
  GenerationSummary,
  LanguageOverview,
  LanguageSummary,
  LLMConnection,
  LLMSettingsView,
  NewConcept,
  Questionnaire,
  SemanticConcept,
  World,
} from "./types";

export class ApiError extends Error {
  constructor(public body: ApiErrorBody["error"]) {
    super(body.message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (response.ok) return response.json();

  let body: ApiErrorBody | null = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON error, e.g. the backend isn't running and the proxy failed.
  }
  throw new ApiError(
    body?.error ?? {
      code: "NETWORK_ERROR",
      message: `The server responded with ${response.status}. Is the backend running on port 8000?`,
      issues: [],
    },
  );
}

export function getQuestionnaire(seed?: number): Promise<Questionnaire> {
  return request(seed === undefined ? "/api/questionnaire" : `/api/questionnaire?seed=${seed}`);
}

export function getConcepts(): Promise<SemanticConcept[]> {
  return request("/api/concepts");
}

export function getLlmSettings(): Promise<LLMSettingsView> {
  return request("/api/settings/llm");
}

export function listLlmModels(connection: LLMConnection): Promise<{ models: string[] }> {
  return request("/api/settings/llm/models", { method: "POST", body: JSON.stringify(connection) });
}

export function saveLlmSettings(connection: LLMConnection & { model: string }): Promise<LLMSettingsView> {
  return request("/api/settings/llm", { method: "PUT", body: JSON.stringify(connection) });
}

export function startGeneration(
  seed: number,
  answers: Answers,
  customAnswers: Record<string, string>,
  newConcepts: NewConcept[],
  derivedRatio: number,
): Promise<{ generation_id: string }> {
  return request("/api/generations", {
    method: "POST",
    body: JSON.stringify({
      seed, answers, custom_answers: customAnswers, new_concepts: newConcepts, derived_ratio: derivedRatio,
    }),
  });
}

export function getGeneration(id: string): Promise<GenerationSummary> {
  return request(`/api/generations/${id}`);
}

export function listLanguages(): Promise<LanguageSummary[]> {
  return request("/api/languages");
}

export function getLanguage(id: string): Promise<LanguageOverview> {
  return request(`/api/languages/${id}`);
}

export function getLanguageWorld(id: string): Promise<World> {
  return request(`/api/languages/${id}/world`);
}

export function getGenerationLog(id: string): Promise<GenerationLog> {
  return request(`/api/languages/${id}/generation`);
}

export function searchDictionary(
  id: string,
  params: { q?: string; pos?: string; origin?: string; page?: number; page_size?: number },
): Promise<EntryPage> {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") query.set(k, String(v));
  return request(`/api/languages/${id}/dictionary?${query}`);
}

export function getEntry(id: string, lexemeId: string): Promise<EntryDetail> {
  return request(`/api/languages/${id}/dictionary/${lexemeId}`);
}
