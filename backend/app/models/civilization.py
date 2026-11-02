from enum import StrEnum

from pydantic import Field, model_validator

from app.models.common import DomainModel, new_id
from app.models.phonology import Phonology


class Civilization(DomainModel):
    id: str = Field(default_factory=new_id)
    name: str
    founding_year: int
    # Technology level (see models/eras.py). None only for languages saved before eras existed.
    era: str | None = None
    geography: str
    economy: str
    social_organization: str | None = None
    cultural_traits: list[str] = Field(default_factory=list)
    description: str | None = None
    # Fields filled in by the generator rather than the user.
    generated_fields: list[str] = Field(default_factory=list)


class Attitude(StrEnum):
    """How one civilization regards another. Drives prestige borrowing later."""

    ADMIRATION = "admiration"
    RESPECT = "respect"
    INDIFFERENCE = "indifference"
    DISTRUST = "distrust"
    CONTEMPT = "contempt"
    HOSTILITY = "hostility"


class RelativePower(StrEnum):
    MUCH_WEAKER = "much_weaker"
    WEAKER = "weaker"
    EQUAL = "equal"
    STRONGER = "stronger"
    MUCH_STRONGER = "much_stronger"

    def inverse(self) -> "RelativePower":
        order = list(RelativePower)
        return order[len(order) - 1 - order.index(self)]


class Separation(StrEnum):
    """What lies between two civilizations; affects how much contact they have."""

    LAND_BORDER = "land_border"
    RIVER = "river"
    SEA = "sea"
    MOUNTAINS = "mountains"
    DESERT = "desert"


class ContactIntensity(StrEnum):
    SPORADIC = "sporadic"
    REGULAR = "regular"
    INTENSE = "intense"


class ContactType(StrEnum):
    TRADE = "trade"
    WARFARE = "warfare"
    RAIDING = "raiding"
    TRIBUTE = "tribute"
    DIPLOMACY = "diplomacy"
    RELIGIOUS = "religious"
    INTERMARRIAGE = "intermarriage"
    MIGRATION = "migration"


class CivilizationRelationship(DomainModel):
    """Facts shared by a pair of civilizations, plus each side's attitude.

    Attitudes are directional so asymmetry ("A admires B, B is indifferent
    to A") is representable. Power is stored from A's point of view.
    """

    id: str = Field(default_factory=new_id)
    civilization_a_id: str
    civilization_b_id: str
    attitude_a_to_b: Attitude
    attitude_b_to_a: Attitude
    power_of_b_relative_to_a: RelativePower
    separation: Separation
    contact_intensity: ContactIntensity
    contact_types: list[ContactType] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_parties(self) -> "CivilizationRelationship":
        if self.civilization_a_id == self.civilization_b_id:
            raise ValueError("a civilization cannot have a relationship with itself")
        return self

    def involves(self, civilization_id: str) -> bool:
        return civilization_id in (self.civilization_a_id, self.civilization_b_id)

    def other(self, civilization_id: str) -> str:
        if civilization_id == self.civilization_a_id:
            return self.civilization_b_id
        if civilization_id == self.civilization_b_id:
            return self.civilization_a_id
        raise KeyError(civilization_id)

    def attitude(self, viewer_id: str, target_id: str) -> Attitude:
        if (viewer_id, target_id) == (self.civilization_a_id, self.civilization_b_id):
            return self.attitude_a_to_b
        if (viewer_id, target_id) == (self.civilization_b_id, self.civilization_a_id):
            return self.attitude_b_to_a
        raise KeyError((viewer_id, target_id))

    def power_of(self, subject_id: str, relative_to_id: str) -> RelativePower:
        if (subject_id, relative_to_id) == (self.civilization_b_id, self.civilization_a_id):
            return self.power_of_b_relative_to_a
        if (subject_id, relative_to_id) == (self.civilization_a_id, self.civilization_b_id):
            return self.power_of_b_relative_to_a.inverse()
        raise KeyError((subject_id, relative_to_id))


class DonorWord(DomainModel):
    """A word in a neighbor's language, created only when something borrows it."""

    id: str = Field(default_factory=new_id)
    concept_id: str
    phonemes: list[str] = Field(min_length=1)
    orthography: str
    # The earlier donor word this one was built from (e.g. "rider" from "horse"),
    # so later loans extend the already-borrowed lexicon instead of starting fresh.
    base_word_id: str | None = None


class LanguageSketch(DomainModel):
    """Minimal stand-in for a neighbor's language.

    The sound system is fixed at creation so lazily generated words stay
    consistent with each other. `words` starts empty and grows only as
    events borrow from this language.
    """

    id: str = Field(default_factory=new_id)
    civilization_id: str
    phonology: Phonology
    words: list[DonorWord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _base_words_exist(self) -> "LanguageSketch":
        ids = {w.id for w in self.words}
        for w in self.words:
            if w.base_word_id is not None and w.base_word_id not in ids:
                raise ValueError(f"donor word {w.id} builds on unknown word {w.base_word_id}")
        return self

    def word_for(self, concept_id: str) -> DonorWord | None:
        return next((w for w in self.words if w.concept_id == concept_id), None)


class World(DomainModel):
    """The focal civilization and its neighbors.

    MVP generates only focal-neighbor relationships; the model allows any pair
    so neighbor-to-neighbor relations can be added later. Neighbors get a
    LanguageSketch; the focal civilization's full Language lives elsewhere.
    """

    id: str = Field(default_factory=new_id)
    seed: int
    focal_civilization_id: str
    civilizations: list[Civilization]
    relationships: list[CivilizationRelationship] = Field(default_factory=list)
    language_sketches: list[LanguageSketch] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_exist(self) -> "World":
        ids = {c.id for c in self.civilizations}
        if self.focal_civilization_id not in ids:
            raise ValueError("focal civilization is not in the world")
        for r in self.relationships:
            if not {r.civilization_a_id, r.civilization_b_id} <= ids:
                raise ValueError(f"relationship {r.id} references an unknown civilization")
        for s in self.language_sketches:
            if s.civilization_id not in ids:
                raise ValueError(f"language sketch {s.id} references an unknown civilization")
        return self

    def language_sketch(self, civilization_id: str) -> LanguageSketch:
        return next(s for s in self.language_sketches if s.civilization_id == civilization_id)

    def civilization(self, civilization_id: str) -> Civilization:
        return next(c for c in self.civilizations if c.id == civilization_id)

    @property
    def focal(self) -> Civilization:
        return self.civilization(self.focal_civilization_id)

    @property
    def neighbors(self) -> list[Civilization]:
        return [c for c in self.civilizations if c.id != self.focal_civilization_id]

    def relationship(self, a_id: str, b_id: str) -> CivilizationRelationship:
        return next(r for r in self.relationships if r.involves(a_id) and r.involves(b_id))
