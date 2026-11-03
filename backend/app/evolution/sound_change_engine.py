"""Deterministic sound-change engine: preview and apply rules to a whole language.

The engine, not the caller, decides which words change: a rule applies to
every word that meets its condition (regular sound change). Callers - later
the evolution agent - choose and tune rules, inspect the preview, and only
then commit.

Application is simultaneous: every condition is checked against the word as
it was before the rule, so k -> tʃ / _ i applied to "kiki" gives "tʃitʃi" in
one pass, and a rule never feeds itself.

Several rules can form one step (one new language version), applied in
order and checked for legality only at the end: final vowel loss may create
"ojenr" as long as a later rule in the same step breaks the cluster up
("ojener"), which is how such repairs happen historically.

Exceptions: a word may skip a rule only through an explicit, explained
`SoundChangeException`, at most MAX_EXCEPTION_SHARE of the dictionary per
step, and only where skipping actually makes a difference.

Morphemes are never rewritten. Every current form is reproducible from
history (`app.evolution.history.realize`), and the lexicon validator checks it.
"""

from collections import defaultdict
from collections.abc import Sequence
from functools import cache

from app.evolution.history import Realized, apply_change, apply_changes, apply_to_word
from app.generation.phoneme_generator import render_phonemes
from app.models.common import DomainModel
from app.models.language import LanguageState, LanguageVersion
from app.models.lexicon import LexemeForm
from app.models.phonology import SUPPORTED_PATTERNS, Phoneme, PhonemeClass, Phonology, pattern_order
from app.models.sound_change import SoundChange, SoundChangeException
from app.models.sounds import SOUNDS, sound
from app.validation.phonology_validator import syllabify, validate_phoneme_sequence

__all__ = ["SoundChangeError", "SoundChangePreview", "SoundChangeResult", "apply_change", "apply_changes",
           "apply_sound_change", "apply_to_word", "resolve_sound"]

# Exceptions per step: at most this share of the dictionary, but always at least a couple.
MAX_EXCEPTION_SHARE = 0.05
MIN_EXCEPTIONS_ALLOWED = 2


class SoundChangeError(ValueError):
    pass


# --- what each sound becomes ------------------------------------------------------

@cache
def _resolve_cached(change_json: str, ipa: str) -> str | None:
    change = SoundChange.model_validate_json(change_json)
    if change.delete:
        return None
    if change.output_mapping is not None:
        return change.output_mapping[ipa]
    return _closest_sound(ipa, change.output_features)


def resolve_sound(change: SoundChange, ipa: str) -> str | None:
    """What /ipa/ becomes under `change` (None = deleted). Only call for sounds the rule targets."""
    return _resolve_cached(change.model_dump_json(), ipa)


# Follows from place and manner (s, ʃ, ts... are sibilants), so it is not compared:
# k -> {place: postalveolar, manner: affricate} should give tʃ even though tʃ is sibilant and k is not.
_DERIVED_FEATURES = {"sibilant"}


def _closest_sound(ipa: str, overrides: dict) -> str:
    """The sound that is /ipa/ with `overrides` applied and every other feature unchanged.

    If no such sound exists (voicing /q/: there is no voiced uvular stop in the catalog), the
    sound is left as it is - the rule simply has nothing to turn it into.
    """
    source = sound(ipa)
    wanted = {k: v for k, v in {**source.feature_values(), **overrides}.items() if k not in _DERIVED_FEATURES}
    matches = [s.ipa for s in SOUNDS.values()
               if all(s.feature_values().get(k) == v for k, v in wanted.items())]
    if len(matches) > 1:
        raise SoundChangeError(f"/{ipa}/ with {overrides} could be any of {matches}; "
                               "use output_mapping to say which")
    return matches[0] if matches else ipa


# --- preview ----------------------------------------------------------------------

class WordChange(DomainModel):
    lexeme_id: str
    concept_id: str
    before: str
    after: str
    before_ipa: str
    after_ipa: str


class Homophones(DomainModel):
    form: str
    concept_ids: list[str]


class IllegalWord(DomainModel):
    concept_id: str
    form: str
    problems: list[str]


class StepSummary(DomainModel):
    """What one rule of the step did."""

    change: SoundChange
    # Sounds the rule turned into something else somewhere (None = deleted).
    sound_mapping: dict[str, str | None]
    inserted: list[str]
    # Sounds the rule targets but has nothing to turn into (e.g. /q/ under voicing), left as they are.
    unchanged_sounds: list[str]
    words_changed: int
    exceptions: list[str]  # concept ids that skipped this rule


class SoundChangePreview(DomainModel):
    changes: list[SoundChange]
    steps: list[StepSummary]
    changed: list[WordChange]
    total_words: int
    new_sounds: list[str]
    lost_sounds: list[str]
    new_homophones: list[Homophones]
    illegal: list[IllegalWord]
    # How the sound system's rules had to change to describe the new words.
    phonotactic_changes: list[str]

    @property
    def ok(self) -> bool:
        return not self.illegal


class SoundChangeResult(DomainModel):
    preview: SoundChangePreview
    state: LanguageState | None  # None when the preview has illegal words


def _spell_new_sound(ipa: str, taken: set[str], spellings: dict[str, str]) -> str:
    for candidate in (spellings.get(ipa), sound(ipa).default_grapheme, ipa if len(ipa) == 1 else None):
        if candidate and candidate not in taken:
            return candidate
    raise SoundChangeError(f"/{ipa}/ needs a spelling: its default {sound(ipa).default_grapheme!r} is already "
                           "used by this language; pass one in `spellings`")


def _evolve_phonology(old: Phonology, new_words: list[list[str]], old_words: list[list[str]],
                      spellings: dict[str, str]) -> tuple[Phonology, list[str]]:
    """The sound system that describes the new words: inventory, spelling, syllable rules."""
    notes: list[str] = []
    before = {p for w in old_words for p in w}
    after = {p for w in new_words for p in w}

    # Inventory: sounds that disappeared from every word are gone; sounds that appeared join.
    kept = [p for p in old.phonemes if p.id in after or p.id not in before]
    taken = {p.grapheme for p in kept}
    added = []
    for ipa in sorted(after - {p.id for p in old.phonemes}):
        grapheme = _spell_new_sound(ipa, taken, spellings)
        taken.add(grapheme)
        added.append(Phoneme.from_ipa(ipa, grapheme=grapheme))
    phonemes = kept + added

    # Hiatus, s-clusters, and syllable shapes: whatever the new words need, if the engine supports it.
    vowel = lambda p: sound(p).phoneme_class == PhonemeClass.VOWEL  # noqa: E731
    hiatus = old.allow_hiatus or any(vowel(a) and vowel(b) for w in new_words for a, b in zip(w, w[1:]))
    if hiatus != old.allow_hiatus:
        notes.append("adjacent vowels are now allowed")

    def system(patterns: list[str], s_clusters: bool) -> Phonology:
        return Phonology(phonemes=phonemes, syllable_patterns=patterns, allow_hiatus=hiatus,
                         allow_s_clusters=s_clusters)

    every_shape = sorted(SUPPORTED_PATTERNS, key=pattern_order)
    s_clusters = old.allow_s_clusters or any(
        w and syllabify(w, system(every_shape, False)) is None and syllabify(w, system(every_shape, True))
        for w in new_words)
    if s_clusters != old.allow_s_clusters:
        notes.append("s may now cluster with stops at syllable edges (st-, -ks)")

    # A shape is added only for a word the existing shapes cannot parse: "aplo" stays ap.lo
    # in a CVC language rather than gaining a CCV it doesn't need.
    patterns = list(old.syllable_patterns)
    for w in new_words:
        if w and syllabify(w, system(patterns, s_clusters)) is None:
            if syllables := syllabify(w, system(every_shape, s_clusters)):
                patterns += sorted({x.pattern for x in syllables} - set(patterns), key=pattern_order)
    if added_shapes := patterns[len(old.syllable_patterns):]:
        notes.append(f"new syllable shapes: {', '.join(added_shapes)}")

    # Position restrictions: keep only those the new words still obey.
    ids = {p.id for p in phonemes}
    restrictions = []
    for r in old.restrictions:
        if r.phoneme_id not in ids:
            continue
        probe = Phonology(phonemes=phonemes, syllable_patterns=patterns, allow_hiatus=hiatus,
                          allow_s_clusters=s_clusters, restrictions=[r])
        if all(validate_phoneme_sequence(w, probe).ok for w in new_words if w):
            restrictions.append(r)
        else:
            notes.append(f"{probe.spell([r.phoneme_id])} may now appear in "
                         f"{', '.join(pos.value for pos in r.forbidden_positions)}")
    return Phonology(phonemes=phonemes, syllable_patterns=patterns, allow_hiatus=hiatus,
                     allow_s_clusters=s_clusters, restrictions=restrictions), notes


def _check_request(state: LanguageState, changes: list[SoundChange], exceptions: list[SoundChangeException],
                   year: int) -> None:
    if not changes:
        raise SoundChangeError("no sound changes given")
    if year <= state.version.valid_from_year:
        raise SoundChangeError(f"year {year} is not after the current version ({state.version.valid_from_year})")
    if len({c.id for c in changes}) != len(changes):
        raise SoundChangeError("two changes in one step share an id")
    known_changes = {c.id for c in changes}
    known_words = {lx.id for lx in state.lexemes}
    for e in exceptions:
        if e.change_id not in known_changes:
            raise SoundChangeError(f"exception for {e.lexeme_id} names a change that is not in this step")
        if e.lexeme_id not in known_words:
            raise SoundChangeError(f"exception names unknown word {e.lexeme_id}")
    limit = max(MIN_EXCEPTIONS_ALLOWED, int(len(state.lexemes) * MAX_EXCEPTION_SHARE))
    if len(exceptions) > limit:
        raise SoundChangeError(f"{len(exceptions)} exceptions is too many: at most {limit} words may skip the "
                               "changes of one step. Sound change is regular; narrow the rule instead")


def apply_sound_change(state: LanguageState, changes: SoundChange | Sequence[SoundChange], year: int,
                       spellings: dict[str, str] | None = None,
                       exceptions: Sequence[SoundChangeException] = ()) -> SoundChangeResult:
    """Preview a step of one or more changes on every word; if every result is legal, build the next version.

    `spellings` chooses letters for sounds the language doesn't have yet (default: the catalog's).
    Nothing is saved: the caller decides whether to commit `result.state`.
    """
    changes = [changes] if isinstance(changes, SoundChange) else list(changes)
    exceptions = list(exceptions)
    _check_request(state, changes, exceptions, year)
    old_ph = state.version.phonology
    concept = {lx.id: lx.concept_id for lx in state.lexemes}
    start = {lx.id: Realized(list(f.phonemes), list(f.segments))
             for lx in state.lexemes for f in [state.form_of(lx.id)]}
    skip = {(e.change_id, e.lexeme_id) for e in exceptions}

    current, steps = dict(start), []
    for change in changes:
        mapping: dict[str, str | None] = {}
        inserted: set[str] = set()
        changed_count, skipped = 0, []
        for lexeme_id, form in current.items():
            out = apply_to_word(change, form.phonemes)
            if all(o == [p] for o, p in zip(out, form.phonemes)):
                if (change.id, lexeme_id) in skip:
                    raise SoundChangeError(f"{concept[lexeme_id]!r} is listed as an exception to "
                                           f"{change.name!r}, but the rule does not change it")
                continue
            if (change.id, lexeme_id) in skip:
                skipped.append(concept[lexeme_id])
                continue
            for p, o in zip(form.phonemes, out):
                if change.insert is not None:
                    extra = list(o)
                    extra.remove(p)  # what is left beside the original sound was inserted
                    inserted.update(extra)
                elif o != [p]:
                    mapping[p] = o[0] if o else None
            current[lexeme_id] = apply_change(change, form)
            changed_count += 1
        unchanged = sorted(p.id for p in old_ph.phonemes
                           if change.output_features is not None and change.targets(p.id)
                           and resolve_sound(change, p.id) == p.id)
        steps.append(StepSummary(change=change, sound_mapping=mapping, inserted=sorted(inserted),
                                 unchanged_sounds=unchanged, words_changed=changed_count, exceptions=skipped))

    lexeme_ids = [lx.id for lx in state.lexemes]
    old_words = [start[i].phonemes for i in lexeme_ids]
    new_words = [current[i].phonemes for i in lexeme_ids]
    new_ph, notes = _evolve_phonology(old_ph, new_words, old_words, spellings or {})

    changed, illegal = [], []
    for lexeme_id, before, after in zip(lexeme_ids, old_words, new_words):
        if after != before:
            changed.append(WordChange(lexeme_id=lexeme_id, concept_id=concept[lexeme_id],
                                      before=state.form_of(lexeme_id).orthography, after=new_ph.spell(after),
                                      before_ipa="".join(before), after_ipa="".join(after)))
        if not after:
            illegal.append(IllegalWord(concept_id=concept[lexeme_id], form="", problems=["every sound is lost"]))
            continue
        check = validate_phoneme_sequence(after, new_ph)
        if not check.ok:
            illegal.append(IllegalWord(concept_id=concept[lexeme_id], form=new_ph.spell(after),
                                       problems=[i.message for i in check.errors]))

    def homophone_groups(words: list[list[str]]) -> dict[tuple[str, ...], list[str]]:
        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for lexeme_id, w in zip(lexeme_ids, words):
            groups[tuple(w)].append(concept[lexeme_id])
        return {k: v for k, v in groups.items() if len(v) > 1}

    old_groups = homophone_groups(old_words)
    new_groups = [Homophones(form=new_ph.spell(k), concept_ids=v)
                  for k, v in homophone_groups(new_words).items() if old_groups.get(k) != v and k]

    old_sounds = {p for w in old_words for p in w}
    new_sounds = {p for w in new_words for p in w}
    preview = SoundChangePreview(
        changes=changes, steps=steps, changed=changed, total_words=len(lexeme_ids),
        new_sounds=sorted(new_sounds - old_sounds), lost_sounds=sorted(old_sounds - new_sounds),
        new_homophones=new_groups, illegal=illegal, phonotactic_changes=notes)
    if not preview.ok:
        return SoundChangeResult(preview=preview, state=None)

    repair = state.version.boundary_repair
    if repair:
        # The linking sounds evolve like any other sound, judged on their own.
        repair = repair.model_copy(update={
            field: (shape[0] if (shape := apply_changes(changes, [p])) and len(shape) == 1 else None) if p else None
            for field, p in (("linking_vowel", repair.linking_vowel), ("linking_consonant", repair.linking_consonant))
        })
        if not (repair.linking_vowel or repair.linking_consonant) or not all(
                p is None or new_ph.has_phoneme(p) for p in (repair.linking_vowel, repair.linking_consonant)):
            repair = None  # new derivations will need a new repair strategy
    version = LanguageVersion(language_id=state.language.id, valid_from_year=year,
                              predecessor_version_id=state.version.id, phonology=new_ph,
                              grammar=state.version.grammar, boundary_repair=repair,
                              sound_changes=changes, exceptions=exceptions)
    new_forms = [LexemeForm(lexeme_id=i, language_version_id=version.id, phonemes=current[i].phonemes,
                            orthography=render_phonemes(current[i].phonemes, new_ph), segments=current[i].segments)
                 for i in lexeme_ids]
    new_state = state.model_copy(update={
        "version": version,
        "previous_versions": [*state.previous_versions, state.version],
        "forms": [*state.forms, *new_forms],
    })
    return SoundChangeResult(preview=preview, state=new_state)
