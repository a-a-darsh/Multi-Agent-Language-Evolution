// Mirrors of the backend Pydantic models (app/models). Keep in sync by hand for now.

export type InputType =
  | "single_select"
  | "multi_select"
  | "free_text"
  | "yes_no"
  | "custom_word"
  | "custom_vocabulary";

export interface QuestionOption {
  value: string;
  label: string;
  description?: string | null;
}

export interface Question {
  id: string;
  category: string;
  prompt: string;
  help?: string | null;
  input_type: InputType;
  options: QuestionOption[];
  required: boolean;
  maps_to: string;
}

export interface Questionnaire {
  seed: number;
  questions: Question[];
}

export interface SemanticConcept {
  id: string;
  gloss: string;
  part_of_speech: string;
  categories: string[];
  priority: "core" | "cultural";
}

export type Answer = string | string[] | boolean | Record<string, string> | undefined;
export type Answers = Record<string, Answer>;

export interface ValidationIssue {
  code: string;
  message: string;
  object_id?: string | null;
  details: Record<string, unknown>;
}

export interface ApiErrorBody {
  error: { code: string; message: string; issues: ValidationIssue[] };
}

export type Attitude = "admiration" | "respect" | "indifference" | "distrust" | "contempt" | "hostility";

export interface Civilization {
  id: string;
  name: string;
  founding_year: number;
  geography: string;
  economy: string;
  social_organization?: string | null;
  cultural_traits: string[];
  description?: string | null;
  generated_fields: string[];
}

export interface CivilizationRelationship {
  id: string;
  civilization_a_id: string;
  civilization_b_id: string;
  attitude_a_to_b: Attitude;
  attitude_b_to_a: Attitude;
  power_of_b_relative_to_a: string;
  separation: string;
  contact_intensity: string;
  contact_types: string[];
}

export interface Phoneme {
  id: string;
  grapheme: string;
  phoneme_class: "C" | "V";
}

export interface LanguageSketch {
  id: string;
  civilization_id: string;
  phonology: { phonemes: Phoneme[]; syllable_patterns: string[] };
  words: unknown[];
}

export interface World {
  id: string;
  seed: number;
  focal_civilization_id: string;
  civilizations: Civilization[];
  relationships: CivilizationRelationship[];
  language_sketches: LanguageSketch[];
}

export type LLMProvider = "local" | "openai" | "anthropic" | "openrouter";

export interface LLMSettingsView {
  configured: boolean;
  provider: LLMProvider | null;
  model: string | null;
  base_url: string | null;
  api_key_hint: string | null;
}

export interface LLMConnection {
  provider: LLMProvider;
  api_key?: string;
  base_url?: string;
}

export interface WorldGenerationResponse {
  request: unknown;
  world: World;
}
