"""Word histories: what every word should look like, replayed from the language's past.

The single source of truth for evolved forms. The sound-change engine, the
lexicon validator, and word coinage all go through it, so they cannot drift
apart.

Replay, version by version (oldest first):
1. Founding version: each founding word is assembled from its morphemes with
   the founding sound system.
2. Each later version: its sound changes run over every existing word in
   order, skipping a word only where an exception was recorded; then the words
   coined during that version are built:
   - a new root from its morpheme, as written when it was coined;
   - a new derivation from its base word *as it sounded then* plus the affix's
     shape then (the founding affix with every change since applied to it on
     its own), joined with that version's boundary rules;
   - a new compound from its two parts as they sounded then, joined the same way.

Rules substitute, delete, or insert single sounds, so each output sound can
be traced to the morpheme it came from: segmentation survives every change.
An inserted sound joins the segment of the sound before it (or after it, at
the start of a word).
"""

from collections.abc import Sequence
from typing import NamedTuple

from app.generation.morphological_engine import MorphologyError, assemble
from app.models.language import LanguageState, LanguageVersion
from app.models.lexicon import FormSegment, Lexeme
from app.models.morphology import Morpheme, MorphemeType
from app.models.phonology import PhonemeClass
from app.models.sound_change import ContextItem, SoundChange
from app.models.sounds import SOUNDS
from app.models.validation import ValidationResult


class Realized(NamedTuple):
    phonemes: list[str]
    segments: list[FormSegment]


# --- one change on one word ----------------------------------------------------------

def _is_consonant(ipa: str) -> bool:
    return SOUNDS[ipa].phoneme_class == PhonemeClass.CONSONANT


def _context_matches(items: Sequence[ContextItem], word: Sequence[str], start: int, step: int) -> bool:
    """Do `items` match `word` starting at `start`, walking in direction `step` (+1 right, -1 left)?"""

    def match(k: int, j: int) -> bool:
        if k == len(items):
            return True
        item = items[k]
        if item.kind == "boundary":
            return (j < 0 or j >= len(word)) and match(k + 1, j)
        if item.kind == "segment":
            return 0 <= j < len(word) and item.matcher.matches(word[j]) and match(k + 1, j + step)
        # consonants: zero or more, shortest first
        while True:
            if match(k + 1, j):
                return True
            if not (0 <= j < len(word) and _is_consonant(word[j])):
                return False
            j += step

    return match(0, start)


def _left_ok(change: SoundChange, word: Sequence[str], before: int) -> bool:
    # Left context is written in reading order, so walk it backwards from the position before.
    return _context_matches(list(reversed(change.left)), word, before, -1)


def apply_to_word(change: SoundChange, word: Sequence[str]) -> list[list[str]]:
    """Per input position, what it becomes: [] deleted, [x] kept/changed, longer with an insertion.

    Every condition is checked against the input word (simultaneous application).
    """
    from app.evolution.sound_change_engine import resolve_sound  # resolution lives with the engine

    if change.insert is not None:
        out = [[p] for p in word]
        for gap in range(len(word) + 1):  # gap g sits between positions g-1 and g
            if _left_ok(change, word, gap - 1) and _context_matches(change.right, word, gap, +1):
                if gap == 0:
                    out[0].insert(0, change.insert)
                else:
                    out[gap - 1].append(change.insert)
        return out
    result = []
    for i, p in enumerate(word):
        if change.targets(p) and _left_ok(change, word, i - 1) and _context_matches(change.right, word, i + 1, +1):
            new = resolve_sound(change, p)
            result.append([] if new is None else [new])
        else:
            result.append([p])
    return result


def apply_change(change: SoundChange, form: Realized) -> Realized:
    out = apply_to_word(change, form.phonemes)
    segments, i = [], 0
    for seg in form.segments:
        kept = [p for pos in out[i:i + len(seg.phonemes)] for p in pos]
        i += len(seg.phonemes)
        if kept:  # a segment whose every sound was lost disappears from the word
            segments.append(FormSegment(phonemes=kept, morpheme_id=seg.morpheme_id))
    return Realized([p for pos in out for p in pos], segments)


def apply_changes(changes: Sequence[SoundChange], word: Sequence[str]) -> list[str]:
    """Run an isolated word through several changes in historical order."""
    form = Realized(list(word), [])
    for change in changes:
        form = apply_change(change, form)
    return form.phonemes


# --- replaying a whole language ------------------------------------------------------

def versions_of(state: LanguageState) -> list[LanguageVersion]:
    return [*state.previous_versions, state.version]


def _version_index(state: LanguageState, version_id: str | None) -> int:
    if version_id is None:
        return 0
    for i, v in enumerate(versions_of(state)):
        if v.id == version_id:
            return i
    raise KeyError(f"unknown language version {version_id}")


def morpheme_shape(state: LanguageState, morpheme: Morpheme, at_index: int | None = None) -> list[str]:
    """How a morpheme sounds on its own at a version: as introduced, plus every change since."""
    versions = versions_of(state)
    at_index = len(versions) - 1 if at_index is None else at_index
    start = _version_index(state, morpheme.introduced_in_version_id)
    changes = [c for v in versions[start + 1:at_index + 1] for c in v.sound_changes]
    return apply_changes(changes, morpheme.phonemes)


def build_coined(state: LanguageState, lexeme: Lexeme, index: int, forms: dict[str, Realized]) -> Realized:
    """A word coined at version `index`, from the shapes of its parts at that time. Raises MorphologyError."""
    version = versions_of(state)[index]
    morphemes = {m.id: m for m in state.morphemes}
    if lexeme.compound is not None:
        parts = [forms[p] for p in lexeme.compound.part_lexeme_ids]
        pieces = [Morpheme(id=f"__part{i}__", language_id=state.language.id, phonemes=p.phonemes,
                           type=MorphemeType.ROOT, gloss="part") for i, p in enumerate(parts)]
        built = assemble(pieces, version.phonology, version.boundary_repair, object_id=lexeme.id,
                         compound_seams=[1])
        segments = [s for seg in built.segments
                    for s in (_part_segments(parts, seg.morpheme_id) or [seg])]
        return Realized(built.phonemes, segments)
    if lexeme.derivation is None:
        pieces = [morphemes[m_id].model_copy(update={"phonemes": morpheme_shape(state, morphemes[m_id], index)})
                  for m_id in lexeme.morpheme_ids]
        built = assemble(pieces, version.phonology, version.boundary_repair, object_id=lexeme.id)
        return Realized(built.phonemes, built.segments)

    base = forms[lexeme.derivation.base_lexeme_id]
    base_piece = Morpheme(id="__base__", language_id=state.language.id, phonemes=base.phonemes,
                          type=MorphemeType.ROOT, gloss="base")
    pieces = [base_piece]
    for m_id in lexeme.derivation.applied_morpheme_ids:
        affix = morphemes[m_id]
        shape = morpheme_shape(state, affix, index)
        if not shape:
            raise MorphologyError(_issue("AFFIX_WORN_AWAY", f"The affix {affix.gloss!r} has no sounds left",
                                         lexeme.id))
        piece = affix.model_copy(update={"phonemes": shape})
        pieces = [piece, *pieces] if affix.type == MorphemeType.PREFIX else [*pieces, piece]
    built = assemble(pieces, version.phonology, version.boundary_repair, object_id=lexeme.id)
    segments = [s for seg in built.segments
                for s in (base.segments or [FormSegment(phonemes=base.phonemes)]
                          if seg.morpheme_id == "__base__" else [seg])]
    return Realized(built.phonemes, segments)


def _compound_seams(lexeme: Lexeme, state: LanguageState) -> list[int]:
    """Where the two words of a founding compound meet, as an index into its morphemes."""
    if lexeme.compound is None:
        return []
    first = next(lx for lx in state.lexemes if lx.id == lexeme.compound.part_lexeme_ids[0])
    return [len(first.morpheme_ids)]


def _part_segments(parts: list[Realized], piece_id: str | None) -> list[FormSegment] | None:
    """A compound part's own segments (its root and affixes), in place of its placeholder piece."""
    if piece_id and piece_id.startswith("__part"):
        part = parts[int(piece_id[len("__part"):-2])]
        return part.segments or [FormSegment(phonemes=part.phonemes)]
    return None


def _issue(code: str, message: str, object_id: str) -> ValidationResult:
    result = ValidationResult()
    result.add(code, message, object_id=object_id)
    return result


def realize(state: LanguageState) -> tuple[dict[str, Realized], ValidationResult]:
    """Every word's expected current form, replayed from history; plus any word that can't be built."""
    result = ValidationResult()
    versions = versions_of(state)
    morphemes = {m.id: m for m in state.morphemes}
    forms: dict[str, Realized] = {}

    for index, version in enumerate(versions):
        if index > 0:
            skip = {(e.change_id, e.lexeme_id) for e in version.exceptions}
            for lexeme_id, form in forms.items():
                for change in version.sound_changes:
                    if (change.id, lexeme_id) not in skip:
                        form = apply_change(change, form)
                forms[lexeme_id] = form

        coined = [lx for lx in state.lexemes
                  if _version_index(state, lx.coined_in_version_id) == index and lx.id not in forms]
        pending = list(coined)
        while pending:  # derived words after their bases
            progress = False
            for lx in list(pending):
                if any(b not in forms and any(p.id == b for p in pending) for b in lx.base_lexeme_ids):
                    continue
                pending.remove(lx)
                progress = True
                if any(m not in morphemes for m in lx.morpheme_ids):
                    continue  # reported by the lexicon validator
                try:
                    if index == 0 and lx.coined_in_version_id is None:  # the founding dictionary
                        built = assemble([morphemes[m] for m in lx.morpheme_ids], version.phonology,
                                         version.boundary_repair, object_id=lx.id,
                                         compound_seams=_compound_seams(lx, state))
                        forms[lx.id] = Realized(built.phonemes, built.segments)
                    elif any(b not in forms for b in lx.base_lexeme_ids):
                        result.add("MISSING_BASE", f"{lx.concept_id!r} was coined from a word that did not exist "
                                   "yet", object_id=lx.id)
                    else:
                        forms[lx.id] = build_coined(state, lx, index, forms)
                except MorphologyError as e:
                    result.extend(e.result)
            if not progress:
                break
    return forms, result
