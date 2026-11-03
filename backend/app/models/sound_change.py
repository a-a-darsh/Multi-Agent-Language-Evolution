"""Sound changes as data: A -> B / left _ right.

A rule says *which* sounds change (`target`), *into what* (`output_features`,
`output_mapping`, or `delete`), and *where* (`left` / `right` context). It
never lists words: the engine applies it to every word that meets the
condition, which is what makes historical sound change regular.

Examples:
    k, ɡ -> tʃ, dʒ before front vowels:
        target={features: {place: velar, manner: stop}},
        output_features={place: postalveolar, manner: affricate},
        right=[segment {class: V, backness: front}]
    vowels drop word-finally in words with an earlier vowel:
        target={features: {class: V}}, delete=True,
        left=[segment {class: V}, consonants], right=[boundary]
    e is inserted to break a final consonant + liquid/nasal (ojenr -> ojener):
        insert="e", left=[segment {class: C}],
        right=[segment {manner: [trill, approximant, nasal]}, boundary]

A feature value in a matcher may be a list, meaning "any of these".

These models are kept LLM-friendly (flat fields, plain feature names), since
the evolution agent will write them.
"""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.models.common import DomainModel, new_id
from app.models.sounds import SOUNDS, Backness, Height, Manner, PhonemeClass, Place

FeatureValue = str | bool
# In matchers a feature may list alternatives: {"manner": ["nasal", "trill"]}.
MatchValue = FeatureValue | list[FeatureValue]

# Every feature a rule may test or set, with its allowed values.
FEATURE_VALUES: dict[str, set[FeatureValue]] = {
    "class": {c.value for c in PhonemeClass},
    "place": {p.value for p in Place},
    "manner": {m.value for m in Manner},
    "voiced": {True, False},
    "lateral": {True, False},
    "sibilant": {True, False},
    "height": {h.value for h in Height},
    "backness": {b.value for b in Backness},
    "rounded": {True, False},
    "long": {True, False},
    "nasal": {True, False},
}


def _same_kind(value: FeatureValue, allowed: set[FeatureValue]) -> bool:
    # True == 1 in Python, so a boolean feature must be given a real bool and vice versa.
    return isinstance(value, bool) == isinstance(next(iter(allowed)), bool)


def _check_features(features: dict, where: str, lists_allowed: bool = False) -> list[str]:
    problems = []
    for key, value in features.items():
        if key not in FEATURE_VALUES:
            problems.append(f"{where}: unknown feature {key!r} (known: {', '.join(FEATURE_VALUES)})")
            continue
        if isinstance(value, list):
            if not lists_allowed:
                problems.append(f"{where}: {key} must be a single value, not a list")
                continue
            if not value:
                problems.append(f"{where}: {key} lists no values")
            values = value
        else:
            values = [value]
        for v in values:
            if not _same_kind(v, FEATURE_VALUES[key]) or v not in FEATURE_VALUES[key]:
                problems.append(f"{where}: {key}={v!r} is not one of {sorted(map(str, FEATURE_VALUES[key]))}")
    return problems


class SoundMatcher(DomainModel):
    """Matches a sound by IPA symbol, by features, or both (all given conditions must hold)."""

    ipa: list[str] | None = None
    features: dict[str, MatchValue] | None = None

    @field_validator("ipa", mode="before")
    @classmethod
    def _one_sound_is_a_list(cls, value):
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check(self) -> "SoundMatcher":
        problems = []
        if not self.ipa and not self.features:
            problems.append("a matcher needs `ipa` or `features`")
        problems += [f"unknown sound /{s}/" for s in self.ipa or [] if s not in SOUNDS]
        problems += _check_features(self.features or {}, "matcher", lists_allowed=True)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def matches(self, ipa: str) -> bool:
        if self.ipa and ipa not in self.ipa:
            return False
        if self.features:
            values = SOUNDS[ipa].feature_values()
            return all(values.get(k) in v if isinstance(v, list) else values.get(k) == v
                       for k, v in self.features.items())
        return True


class ContextItem(DomainModel):
    """One step of a rule's environment.

    segment:    exactly one sound matching `matcher`
    boundary:   the edge of the word (# in linguistic notation)
    consonants: zero or more consonants, e.g. "the next vowel" = [consonants, segment V]
    """

    kind: Literal["segment", "boundary", "consonants"]
    matcher: SoundMatcher | None = None

    @model_validator(mode="after")
    def _check(self) -> "ContextItem":
        if (self.kind == "segment") != (self.matcher is not None):
            raise ValueError("a `segment` context item needs a matcher; other kinds take none")
        return self


class SoundChange(DomainModel):
    id: str = Field(default_factory=new_id)
    name: str
    description: str | None = None
    # Sounds that change. May be omitted when `output_mapping` lists them; not used with `insert`.
    target: SoundMatcher | None = None
    # Exactly one way to say what happens:
    output_features: dict[str, FeatureValue] | None = None  # change these features, keep the rest
    output_mapping: dict[str, str | None] | None = None     # explicit: {"b": "β", "h": None}; None deletes
    delete: bool = False
    # Insertion (epenthesis): this sound appears wherever `left` meets `right`.
    insert: str | None = None
    # Environment, written in reading order: left=[V, consonants] means "after a vowel, any consonants between".
    # For an insertion, left is what comes before the new sound and right what comes after it.
    left: list[ContextItem] = Field(default_factory=list)
    right: list[ContextItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "SoundChange":
        problems = []
        outputs = [self.output_features is not None, self.output_mapping is not None, self.delete,
                   self.insert is not None]
        if sum(outputs) > 1:
            problems.append("a rule does one thing: give only one of output_features, output_mapping, delete, "
                            "or insert. To do two things (e.g. nasalize a vowel and drop the n after it), use "
                            "two rules in the same step; they apply in order")
        elif sum(outputs) == 0:
            problems.append("say what happens: give one of output_features, output_mapping, delete, or insert")
        if self.insert is not None:
            if self.insert not in SOUNDS:
                problems.append(f"insert: unknown sound /{self.insert}/")
            if self.target is not None:
                problems.append("an insertion has no target; say where it happens with left and right")
            if not self.left and not self.right:
                problems.append("an insertion needs a left or right context, or it would apply everywhere")
        elif self.target is None and self.output_mapping is None:
            problems.append("a target is required unless output_mapping lists the sounds")
        if self.output_features is not None:
            if not self.output_features:
                problems.append("output_features is empty")
            if "class" in self.output_features:
                problems.append("output_features cannot change a sound's class (consonant/vowel)")
            problems += _check_features(self.output_features, "output_features")
        for src, dst in (self.output_mapping or {}).items():
            for s in (src, dst):
                if s is not None and s not in SOUNDS:
                    problems.append(f"output_mapping: unknown sound /{s}/")
            if dst is not None and src in SOUNDS and dst in SOUNDS and \
                    SOUNDS[src].phoneme_class != SOUNDS[dst].phoneme_class:
                problems.append(f"output_mapping: /{src}/ -> /{dst}/ changes consonant <-> vowel")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def targets(self, ipa: str) -> bool:
        if self.insert is not None:
            return False
        if self.output_mapping is not None and ipa not in self.output_mapping:
            return False
        return self.target is None or self.target.matches(ipa)


class SoundChangeException(DomainModel):
    """One word that did not take part in a sound change, and why.

    Regular change is the rule; exceptions are a small, explained channel for
    what real languages also do: a change that spread word by word and never
    finished (lexical diffusion), very frequent words resisting or wearing down
    differently, or a word pulled back into line with its family (analogy).
    """

    change_id: str
    lexeme_id: str
    reason: str = Field(min_length=3)
