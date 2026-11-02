"""Read-only views of a stored language for the API: overview, entry lists, entry details.

Everything shown is derived from the structured LanguageState, so the
dictionary explains the system (segments, bases, families) rather than just
listing translations.
"""

from typing import Literal

from pydantic import Field

from app.models.common import DomainModel
from app.models.language import LanguageState
from app.models.lexicon import Lexeme, LexemeOrigin, PartOfSpeech
from app.models.morphology import MorphemeType
from app.validation.phonology_validator import syllabify


class EntrySummary(DomainModel):
    id: str
    word: str
    gloss: str
    concept_id: str
    part_of_speech: PartOfSpeech
    origin: LexemeOrigin
    # Every concept this word expresses (more than one for a shared word).
    concept_ids: list[str] = Field(default_factory=list)


class Segment(DomainModel):
    text: str
    kind: Literal["root", "prefix", "suffix", "linking"]
    gloss: str | None = None


class RuleInfo(DomainModel):
    id: str
    name: str
    meaning: str


class EntryDetail(EntrySummary):
    segments: list[Segment]
    syllables: list[str]
    derived_from: EntrySummary | None = None
    rule: RuleInfo | None = None
    derived_words: list[EntrySummary] = Field(default_factory=list)
    family_root: EntrySummary
    family: list[EntrySummary] = Field(default_factory=list)


class EntryPage(DomainModel):
    total: int
    page: int
    page_size: int
    entries: list[EntrySummary]


class DictionaryView:
    def __init__(self, state: LanguageState):
        self.state = state
        self.lexemes = {lx.id: lx for lx in state.lexemes}
        self.morphemes = {m.id: m for m in state.morphemes}
        self.rules = {r.id: r for r in state.derivational_rules}
        self.children: dict[str, list[Lexeme]] = {}
        for lx in state.lexemes:
            if lx.derivation:
                self.children.setdefault(lx.derivation.base_lexeme_id, []).append(lx)

    def summary(self, lx: Lexeme) -> EntrySummary:
        # A word with several meanings (polysemy) lists them all: "zinc; wealth".
        return EntrySummary(id=lx.id, word=self.state.form_of(lx.id).orthography,
                            gloss="; ".join(s.gloss for s in lx.senses),
                            concept_id=lx.concept_id, part_of_speech=lx.part_of_speech, origin=lx.origin,
                            concept_ids=[s.concept_id for s in lx.senses])

    def search(self, q: str = "", pos: PartOfSpeech | None = None, origin: LexemeOrigin | None = None,
               page: int = 1, page_size: int = 50) -> EntryPage:
        q = q.strip().lower()
        matches = [
            self.summary(lx) for lx in self.state.lexemes
            if (pos is None or lx.part_of_speech == pos) and (origin is None or lx.origin == origin)
        ]
        if q:  # search both directions: native word or English meaning
            matches = [e for e in matches if q in e.word or q in e.gloss.lower() or any(q in c for c in e.concept_ids)]
            matches.sort(key=lambda e: (e.word != q and q not in e.gloss.lower().split("; "), e.word))
        else:
            matches.sort(key=lambda e: e.gloss.lower().removeprefix("to "))
        start = (page - 1) * page_size
        return EntryPage(total=len(matches), page=page, page_size=page_size, entries=matches[start:start + page_size])

    def root_of(self, lx: Lexeme) -> Lexeme:
        while lx.derivation:
            lx = self.lexemes[lx.derivation.base_lexeme_id]
        return lx

    def family(self, root: Lexeme) -> list[Lexeme]:
        family = [root]
        for member in family:
            family.extend(self.children.get(member.id, []))
        return family

    def detail(self, lexeme_id: str) -> EntryDetail | None:
        lx = self.lexemes.get(lexeme_id)
        if lx is None:
            return None
        form = self.state.form_of(lx.id)
        segments = []
        for seg in form.segments:
            text = "".join(seg.phonemes)
            if seg.morpheme_id is None:
                segments.append(Segment(text=text, kind="linking"))
                continue
            m = self.morphemes[seg.morpheme_id]
            if m.type == MorphemeType.ROOT:
                # A root's gloss is the meaning of the word it came from.
                segments.append(Segment(text=text, kind="root", gloss=m.gloss))
            else:
                rule_name = next((r.name for r in self.rules.values() if r.morpheme_id == m.id), None)
                segments.append(Segment(text=text, kind=m.type.value, gloss=m.gloss or rule_name))
        syllables = syllabify(form.phonemes, self.state.version.phonology) or []
        root = self.root_of(lx)
        rule = self.rules.get(lx.derivation.rule_id) if lx.derivation else None
        return EntryDetail(
            **self.summary(lx).model_dump(),
            segments=segments,
            syllables=["".join(s.phonemes) for s in syllables],
            derived_from=self.summary(self.lexemes[lx.derivation.base_lexeme_id]) if lx.derivation else None,
            rule=RuleInfo(id=rule.id, name=rule.name, meaning=rule.meaning) if rule else None,
            derived_words=[self.summary(c) for c in self.children.get(lx.id, [])],
            family_root=self.summary(root),
            family=[self.summary(m) for m in self.family(root)],
        )
