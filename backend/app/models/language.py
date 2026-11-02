from pydantic import Field

from app.models.civilization import Civilization
from app.models.common import DomainModel, new_id
from app.models.grammar import Grammar
from app.models.lexicon import Lexeme, LexemeForm
from app.models.morphology import BoundaryRepair, DerivationalRule, Morpheme
from app.models.phonology import Phonology


class Language(DomainModel):
    id: str = Field(default_factory=new_id)
    civilization_id: str
    name: str
    founding_year: int
    # Set for daughter languages once divergence exists (post-MVP).
    parent_language_id: str | None = None
    initial_generation_id: str | None = None


class LanguageVersion(DomainModel):
    """The language as it stood from `valid_from_year` onward.

    MVP 1 creates only the founding version. Historical simulation will
    append new versions linked by `predecessor_version_id`, never mutate old
    ones. Phonology and grammar are embedded here in memory; persistence may
    normalize them into their own tables.
    """

    id: str = Field(default_factory=new_id)
    language_id: str
    valid_from_year: int
    predecessor_version_id: str | None = None
    phonology: Phonology
    grammar: Grammar
    boundary_repair: BoundaryRepair | None = None


class LanguageState(DomainModel):
    """Everything needed to inspect a language at one version."""

    civilization: Civilization
    language: Language
    version: LanguageVersion
    summary: str | None = None
    morphemes: list[Morpheme] = Field(default_factory=list)
    derivational_rules: list[DerivationalRule] = Field(default_factory=list)
    lexemes: list[Lexeme] = Field(default_factory=list)
    forms: list[LexemeForm] = Field(default_factory=list)

    def morpheme(self, morpheme_id: str) -> Morpheme:
        return next(m for m in self.morphemes if m.id == morpheme_id)

    def lexeme(self, lexeme_id: str) -> Lexeme:
        return next(lx for lx in self.lexemes if lx.id == lexeme_id)

    def form_of(self, lexeme_id: str) -> LexemeForm:
        return next(
            f for f in self.forms
            if f.lexeme_id == lexeme_id and f.language_version_id == self.version.id
        )
