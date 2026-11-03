// Mirrors of the backend Pydantic models (app/models). Keep in sync by hand for now.

export type InputType =
  | "single_select"
  | "multi_select"
  | "free_text"
  | "short_text"
  | "yes_no"
  | "custom_word"
  | "custom_vocabulary";

/** A meaning the user invents in the vocabulary editor, with their word for it. */
export interface NewConcept {
  gloss: string;
  part_of_speech: string;
  word: string;
}

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
  always_ask: boolean;
  allow_custom: boolean;
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
  era?: string | null;
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

export interface GenerationSummary {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  current_step: string | null;
  steps: string[];
  language_id: string | null;
  errors: ValidationIssue[];
}

export interface LanguageSummary {
  id: string;
  name: string;
  civilization_name: string;
  entry_count: number;
  derived_count: number;
  created_at: string;
}

export type Origin = "user_defined" | "generated_root" | "derived" | "compound";

export interface EntrySummary {
  id: string;
  word: string;
  gloss: string;
  concept_id: string;
  part_of_speech: string;
  origin: Origin;
  concept_ids: string[];
}

export interface EntryDetail extends EntrySummary {
  segments: { text: string; kind: "root" | "prefix" | "suffix" | "linking"; gloss: string | null }[];
  syllables: string[];
  derived_from: EntrySummary | null;
  rule: { id: string; name: string; meaning: string } | null;
  compound_of: EntrySummary[];
  derived_words: EntrySummary[];
  family_root: EntrySummary;
  family: EntrySummary[];
}

export interface EntryPage {
  total: number;
  page: number;
  page_size: number;
  entries: EntrySummary[];
}

export interface Sound {
  letter: string;
  ipa: string;
  common: boolean;
}

export interface LanguageOverview {
  id: string;
  name: string;
  summary: string | null;
  civilization: Civilization;
  consonants: Sound[];
  vowels: Sound[];
  syllable_patterns: string[];
  allow_hiatus: boolean;
  restrictions: { letter: string; not_allowed: string[] }[];
  word_order: string;
  morphological_type: string;
  adposition_type: string | null;
  features: { name: string; description: string; realization: string }[];
  affixes: {
    rule_id: string;
    name: string;
    meaning: string;
    form: string;
    position: string;
    takes: string[];
    makes: string;
    examples: EntrySummary[];
  }[];
  linking_vowel: string | null;
  linking_consonant: string | null;
  counts: Record<Origin, number>;
  warnings: ValidationIssue[];
  generation_id: string;
}

export interface GenerationLog {
  seed: number;
  steps: { agent: string; step: string; issues: ValidationIssue[] }[];
  llm_calls: { agent: string; step: string; model: string; seconds: number; error: string | null }[];
}

export interface WorldGenerationResponse {
  request: unknown;
  world: World;
}
