from pydantic import Field

from app.models.common import DomainModel
from app.models.grammar import Grammar
from app.models.morphology import BoundaryRepair, DerivationalRule, Morpheme
from app.models.phonology import Phonology


class LanguageDesign(DomainModel):
    """A validated language specification: everything except the dictionary.

    Produced by the Language Design Agent; the dictionary is built from it in Phase 5.
    """

    language_id: str
    name: str
    phonology: Phonology
    grammar: Grammar
    affixes: list[Morpheme] = Field(default_factory=list)
    derivational_rules: list[DerivationalRule] = Field(default_factory=list)
    boundary_repair: BoundaryRepair | None = None
    summary: str
