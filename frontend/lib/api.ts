import type {
  Answers,
  ApiErrorBody,
  LLMConnection,
  LLMSettingsView,
  Questionnaire,
  SemanticConcept,
  WorldGenerationResponse,
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

export function generateWorld(seed: number, answers: Answers): Promise<WorldGenerationResponse> {
  return request("/api/worlds", { method: "POST", body: JSON.stringify({ seed, answers }) });
}
