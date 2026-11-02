"""Historical events: reusable templates, reusable effects, and recorded occurrences.

A template like "[HOME_CIV] declares war on [CIV1]" is defined once in the
catalog. Each time it happens, an EventRecord binds its slots to real
civilizations and logs exactly what changed.
"""

import re
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from app.models.civilization import Attitude, ContactIntensity, ContactType
from app.models.common import DomainModel, new_id

HOME_CIV = "HOME_CIV"
SLOT_NAME_PATTERN = r"^CIV\d+$"
_PLACEHOLDER = re.compile(r"\[([A-Z_0-9]+)\]")

# Ordered worst to best; "shift" effects move along this scale.
ATTITUDE_SCALE: list[Attitude] = [
    Attitude.HOSTILITY,
    Attitude.CONTEMPT,
    Attitude.DISTRUST,
    Attitude.INDIFFERENCE,
    Attitude.RESPECT,
    Attitude.ADMIRATION,
]


# --- world effects -----------------------------------------------------------

def _require_home_civ(effect_id: str, slots: set[str]) -> None:
    # Only home-neighbor relationships exist; neighbor-neighbor events are out of scope.
    if HOME_CIV not in slots:
        raise ValueError(f"effect {effect_id!r} must involve {HOME_CIV}")


class _WorldEffectBase(DomainModel):
    id: str
    description: str

    @property
    def slots(self) -> set[str]:
        raise NotImplementedError

    @model_validator(mode="after")
    def _involves_home_civ(self):
        _require_home_civ(self.id, self.slots)
        return self


class SetAttitude(_WorldEffectBase):
    kind: Literal["set_attitude"]
    viewer: str
    target: str
    attitude: Attitude
    # Also apply target -> viewer.
    reciprocal: bool = True

    @property
    def slots(self) -> set[str]:
        return {self.viewer, self.target}


class ShiftAttitude(_WorldEffectBase):
    kind: Literal["shift_attitude"]
    viewer: str
    target: str
    # Steps along ATTITUDE_SCALE; positive improves. Clamped at the ends.
    steps: int
    reciprocal: bool = True

    @field_validator("steps")
    @classmethod
    def _nonzero(cls, steps: int) -> int:
        if steps == 0:
            raise ValueError("steps must be non-zero")
        return steps

    @property
    def slots(self) -> set[str]:
        return {self.viewer, self.target}


class AddContactType(_WorldEffectBase):
    kind: Literal["add_contact_type"]
    a: str
    b: str
    contact_type: ContactType

    @property
    def slots(self) -> set[str]:
        return {self.a, self.b}


class RemoveContactType(_WorldEffectBase):
    kind: Literal["remove_contact_type"]
    a: str
    b: str
    contact_type: ContactType

    @property
    def slots(self) -> set[str]:
        return {self.a, self.b}


class SetContactIntensity(_WorldEffectBase):
    kind: Literal["set_contact_intensity"]
    a: str
    b: str
    intensity: ContactIntensity

    @property
    def slots(self) -> set[str]:
        return {self.a, self.b}


WorldEffect = Annotated[
    SetAttitude | ShiftAttitude | AddContactType | RemoveContactType | SetContactIntensity,
    Field(discriminator="kind"),
]


# --- linguistic effects (data only; executed by the future loanword system) ---

class BorrowVocabulary(DomainModel):
    id: str
    description: str
    kind: Literal["borrow_vocabulary"]
    donor: str
    recipient: str
    # Semantic inventory categories the borrowed words come from, e.g. ["commerce"].
    semantic_categories: list[str] = Field(min_length=1)
    max_words: int = Field(default=3, ge=1)

    @property
    def slots(self) -> set[str]:
        return {self.donor, self.recipient}

    @model_validator(mode="after")
    def _involves_home_civ(self) -> "BorrowVocabulary":
        _require_home_civ(self.id, self.slots)
        return self


# Will become a discriminated union as kinds such as sound changes are added.
LinguisticEffect = BorrowVocabulary


# --- templates and catalog ---------------------------------------------------

class EventSlot(DomainModel):
    name: str = Field(pattern=SLOT_NAME_PATTERN)
    description: str | None = None
    # Civilization field -> allowed values, e.g. {"economy": ["mercantile"]}.
    # For list fields like cultural_traits, any overlap matches.
    conditions: dict[str, list[str]] = Field(default_factory=dict)


class EventTemplate(DomainModel):
    id: str
    name: str
    description_template: str
    slots: list[EventSlot] = Field(default_factory=list)
    world_effect_ids: list[str] = Field(default_factory=list)
    linguistic_effect_ids: list[str] = Field(default_factory=list)

    @property
    def slot_names(self) -> set[str]:
        return {HOME_CIV} | {s.name for s in self.slots}

    @model_validator(mode="after")
    def _placeholders_declared(self) -> "EventTemplate":
        names = [s.name for s in self.slots]
        if len(set(names)) != len(names):
            raise ValueError(f"event {self.id!r} declares a slot twice")
        undeclared = set(_PLACEHOLDER.findall(self.description_template)) - self.slot_names
        if undeclared:
            raise ValueError(f"event {self.id!r} uses undeclared placeholders {sorted(undeclared)}")
        return self

    def render(self, names: dict[str, str]) -> str:
        return _PLACEHOLDER.sub(lambda m: names[m.group(1)], self.description_template)


class EventCatalog(DomainModel):
    world_effects: list[WorldEffect] = Field(default_factory=list)
    linguistic_effects: list[LinguisticEffect] = Field(default_factory=list)
    events: list[EventTemplate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_valid(self) -> "EventCatalog":
        for group in (self.world_effects, self.linguistic_effects, self.events):
            ids = [item.id for item in group]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate ids in catalog: {sorted({i for i in ids if ids.count(i) > 1})}")

        world = {e.id: e for e in self.world_effects}
        linguistic = {e.id: e for e in self.linguistic_effects}
        for template in self.events:
            for ref_ids, known, label in ((template.world_effect_ids, world, "world"),
                                          (template.linguistic_effect_ids, linguistic, "linguistic")):
                for effect_id in ref_ids:
                    if effect_id not in known:
                        raise ValueError(f"event {template.id!r} references unknown {label} effect {effect_id!r}")
                    missing = known[effect_id].slots - template.slot_names
                    if missing:
                        raise ValueError(
                            f"event {template.id!r} uses effect {effect_id!r}, which needs slots "
                            f"{sorted(missing)} the event does not declare"
                        )
        return self

    def event(self, template_id: str) -> EventTemplate:
        return next(e for e in self.events if e.id == template_id)

    def world_effect(self, effect_id: str) -> WorldEffect:
        return next(e for e in self.world_effects if e.id == effect_id)


# --- recorded history --------------------------------------------------------

class AppliedChange(DomainModel):
    """One concrete state change, kept so history is explainable."""

    effect_id: str
    relationship_id: str
    field: str
    before: str
    after: str


class EventRecord(DomainModel):
    """An event that actually happened. Append-only history."""

    id: str = Field(default_factory=new_id)
    template_id: str
    year: int
    # Slot name -> civilization id, including HOME_CIV.
    bindings: dict[str, str]
    description: str
    world_changes: list[AppliedChange] = Field(default_factory=list)
    # Linguistic effects triggered but not yet executed (loanword system is future work).
    pending_linguistic_effect_ids: list[str] = Field(default_factory=list)
