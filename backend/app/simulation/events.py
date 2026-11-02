"""Deterministic execution of historical events against a World.

Deciding *which* event happens is the Simulation Director's job (future LLM
agent). This module only checks that a chosen event and its slot bindings
are valid, then applies the world effects and records every change.
"""

import json
from functools import cache

from app.data import DATA_DIR
from app.models.civilization import Civilization, CivilizationRelationship, World
from app.models.events import (
    ATTITUDE_SCALE,
    HOME_CIV,
    AddContactType,
    AppliedChange,
    EventCatalog,
    EventRecord,
    EventSlot,
    EventTemplate,
    RemoveContactType,
    SetAttitude,
    SetContactIntensity,
    ShiftAttitude,
    WorldEffect,
)


class EventError(ValueError):
    pass


@cache
def load_event_catalog() -> EventCatalog:
    raw = json.loads((DATA_DIR / "events.json").read_text(encoding="utf-8"))
    return EventCatalog.model_validate(raw)


def slot_matches(civilization: Civilization, slot: EventSlot) -> bool:
    for field, allowed in slot.conditions.items():
        value = getattr(civilization, field)
        matched = bool(set(value) & set(allowed)) if isinstance(value, list) else value in allowed
        if not matched:
            return False
    return True


def slot_candidates(world: World, template: EventTemplate, slot_name: str,
                    already_bound: set[str] = frozenset()) -> list[Civilization]:
    """Neighbors that could fill a slot. Useful for the Director to pick from."""
    slot = next(s for s in template.slots if s.name == slot_name)
    return [c for c in world.neighbors if c.id not in already_bound and slot_matches(c, slot)]


def _resolve_bindings(world: World, template: EventTemplate, bindings: dict[str, str]) -> dict[str, str]:
    if HOME_CIV in bindings and bindings[HOME_CIV] != world.focal_civilization_id:
        raise EventError("HOME_CIV always refers to the focal civilization")
    resolved = {**bindings, HOME_CIV: world.focal_civilization_id}

    declared = template.slot_names
    if missing := declared - set(resolved):
        raise EventError(f"event {template.id!r} needs slots {sorted(missing)}")
    if extra := set(resolved) - declared:
        raise EventError(f"event {template.id!r} has no slots {sorted(extra)}")
    if len(set(resolved.values())) != len(resolved):
        raise EventError("each slot must be a different civilization")

    known = {c.id for c in world.civilizations}
    for slot in template.slots:
        civ_id = resolved[slot.name]
        if civ_id not in known:
            raise EventError(f"slot {slot.name} refers to unknown civilization {civ_id!r}")
        if not slot_matches(world.civilization(civ_id), slot):
            raise EventError(f"{world.civilization(civ_id).name} does not meet the conditions for {slot.name}: "
                             f"{slot.conditions}")
    return resolved


def _relationship(world: World, a_id: str, b_id: str) -> CivilizationRelationship:
    try:
        return world.relationship(a_id, b_id)
    except StopIteration:
        raise EventError(f"no relationship exists between {world.civilization(a_id).name} "
                         f"and {world.civilization(b_id).name}") from None


def _attitude_field(r: CivilizationRelationship, viewer_id: str) -> str:
    return "attitude_a_to_b" if viewer_id == r.civilization_a_id else "attitude_b_to_a"


def _apply(world: World, effect: WorldEffect, bindings: dict[str, str]) -> list[AppliedChange]:
    changes: list[AppliedChange] = []

    def set_field(r: CivilizationRelationship, field: str, value) -> None:
        before = getattr(r, field)
        if before != value:
            setattr(r, field, value)
            changes.append(AppliedChange(effect_id=effect.id, relationship_id=r.id, field=field,
                                         before=str(before), after=str(value)))

    match effect:
        case SetAttitude() | ShiftAttitude():
            viewer, target = bindings[effect.viewer], bindings[effect.target]
            r = _relationship(world, viewer, target)
            directions = [viewer, target] if effect.reciprocal else [viewer]
            for v in directions:
                field = _attitude_field(r, v)
                if isinstance(effect, SetAttitude):
                    new = effect.attitude
                else:
                    index = ATTITUDE_SCALE.index(getattr(r, field)) + effect.steps
                    new = ATTITUDE_SCALE[max(0, min(index, len(ATTITUDE_SCALE) - 1))]
                set_field(r, field, new)
        case AddContactType() | RemoveContactType():
            r = _relationship(world, bindings[effect.a], bindings[effect.b])
            types = list(r.contact_types)
            if isinstance(effect, AddContactType) and effect.contact_type not in types:
                types.append(effect.contact_type)
            elif isinstance(effect, RemoveContactType) and effect.contact_type in types:
                types.remove(effect.contact_type)
            if types != r.contact_types:
                if not types:
                    raise EventError(f"effect {effect.id!r} would leave no contact at all")
                changes.append(AppliedChange(effect_id=effect.id, relationship_id=r.id, field="contact_types",
                                             before=",".join(r.contact_types), after=",".join(types)))
                r.contact_types = types
        case SetContactIntensity():
            r = _relationship(world, bindings[effect.a], bindings[effect.b])
            set_field(r, "contact_intensity", effect.intensity)
    return changes


def apply_event(world: World, template_id: str, bindings: dict[str, str], year: int,
                catalog: EventCatalog | None = None) -> tuple[World, EventRecord]:
    """Apply an event to a copy of `world`. The input world is never modified."""
    catalog = catalog or load_event_catalog()
    try:
        template = catalog.event(template_id)
    except StopIteration:
        raise EventError(f"unknown event {template_id!r}") from None

    resolved = _resolve_bindings(world, template, bindings)
    new_world = world.model_copy(deep=True)
    changes = [change for effect_id in template.world_effect_ids
               for change in _apply(new_world, catalog.world_effect(effect_id), resolved)]

    record = EventRecord(
        template_id=template.id,
        year=year,
        bindings=resolved,
        description=template.render({slot: world.civilization(cid).name for slot, cid in resolved.items()}),
        world_changes=changes,
        pending_linguistic_effect_ids=list(template.linguistic_effect_ids),
    )
    # Re-validate so a bad effect can't leave the world in an invalid state.
    return World.model_validate(new_world.model_dump()), record
