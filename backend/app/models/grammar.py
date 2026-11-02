from enum import StrEnum

from pydantic import Field

from app.models.common import DomainModel, new_id


class WordOrder(StrEnum):
    SOV = "SOV"
    SVO = "SVO"
    VSO = "VSO"
    VOS = "VOS"
    OVS = "OVS"
    OSV = "OSV"


class MorphologicalType(StrEnum):
    ISOLATING = "isolating"
    AGGLUTINATIVE = "agglutinative"
    FUSIONAL = "fusional"
    POLYSYNTHETIC = "polysynthetic"


class AdpositionType(StrEnum):
    PREPOSITIONS = "prepositions"
    POSTPOSITIONS = "postpositions"


class GrammaticalFeature(DomainModel):
    """A notable grammatical property, e.g. evidentiality or noun classes.

    `realization` describes how the feature is expressed (e.g. "verb suffix").
    `reference_ids` records which RAG documents informed this decision.
    """

    id: str
    name: str
    description: str
    realization: str
    reference_ids: list[str] = Field(default_factory=list)


class Grammar(DomainModel):
    id: str = Field(default_factory=new_id)
    word_order: WordOrder
    morphological_type: MorphologicalType
    adposition_type: AdpositionType | None = None
    features: list[GrammaticalFeature] = Field(default_factory=list)
