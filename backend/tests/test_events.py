import pytest
from pydantic import ValidationError

from app.data import load_semantic_inventory
from app.models.civilization import (
    Attitude,
    Civilization,
    CivilizationRelationship,
    ContactIntensity,
    ContactType,
    RelativePower,
    Separation,
    World,
)
from app.models.events import EventCatalog
from app.simulation.events import EventError, apply_event, load_event_catalog, slot_candidates


def make_world() -> World:
    home = Civilization(id="home", name="Savari", founding_year=0, geography="coastal", economy="mercantile")
    rival = Civilization(id="rival", name="Kalilo", founding_year=0, geography="coastal", economy="mercantile",
                         cultural_traits=["commercial"])
    nomads = Civilization(id="nomads", name="Mido", founding_year=0, geography="steppe",
                          economy="pastoral_nomadic", cultural_traits=["religious"])

    def rel(other, to_other, to_home, types):
        return CivilizationRelationship(
            civilization_a_id="home", civilization_b_id=other, attitude_a_to_b=to_other,
            attitude_b_to_a=to_home, power_of_b_relative_to_a=RelativePower.EQUAL,
            separation=Separation.LAND_BORDER, contact_intensity=ContactIntensity.REGULAR,
            contact_types=types,
        )

    return World(seed=1, focal_civilization_id="home", civilizations=[home, rival, nomads], relationships=[
        rel("rival", Attitude.ADMIRATION, Attitude.RESPECT, [ContactType.TRADE]),
        rel("nomads", Attitude.DISTRUST, Attitude.INDIFFERENCE, [ContactType.RAIDING]),
    ])


def test_catalog_loads_and_borrowing_categories_exist():
    catalog = load_event_catalog()
    categories = {cat for c in load_semantic_inventory() for cat in c.categories}
    for effect in catalog.linguistic_effects:
        assert set(effect.semantic_categories) <= categories, effect.id


def test_war_and_alliance_example():
    world = make_world()
    new_world, record = apply_event(world, "declare_war_and_ally", {"CIV1": "rival", "CIV2": "nomads"}, year=-412)

    war = new_world.relationship("home", "rival")
    assert war.attitude("home", "rival") == war.attitude("rival", "home") == Attitude.HOSTILITY
    assert war.contact_types == [ContactType.WARFARE]  # trade ended, war added

    alliance = new_world.relationship("home", "nomads")
    assert alliance.attitude("home", "nomads") == Attitude.RESPECT        # distrust +2
    assert alliance.attitude("nomads", "home") == Attitude.ADMIRATION     # indifference +2
    assert ContactType.DIPLOMACY in alliance.contact_types

    assert record.description == "Savari declares war on Kalilo and allies with Mido"
    assert record.bindings == {"CIV1": "rival", "CIV2": "nomads", "HOME_CIV": "home"}
    assert record.year == -412
    assert record.pending_linguistic_effect_ids == []
    changed = {(c.effect_id, c.field, c.before, c.after) for c in record.world_changes}
    assert ("home_civ1_become_hostile", "attitude_a_to_b", "admiration", "hostility") in changed
    assert ("home_civ1_end_trade", "contact_types", "trade,warfare", "warfare") in changed


def test_original_world_is_untouched():
    world = make_world()
    before = world.model_dump()
    apply_event(world, "declare_war_and_ally", {"CIV1": "rival", "CIV2": "nomads"}, year=0)
    assert world.model_dump() == before


def test_one_sided_shift_and_pending_loanwords():
    new_world, record = apply_event(make_world(), "foreign_merchants_prosper", {"CIV1": "rival"}, year=0)
    r = new_world.relationship("home", "rival")
    assert r.attitude("home", "rival") == Attitude.ADMIRATION  # already at the top: clamped
    assert r.attitude("rival", "home") == Attitude.RESPECT     # not reciprocal
    assert record.pending_linguistic_effect_ids == ["borrow_commerce_from_civ1"]


def test_attitude_shifts_clamp_at_bottom():
    world, _ = apply_event(make_world(), "declare_war_and_ally", {"CIV1": "rival", "CIV2": "nomads"}, year=0)
    world, record = apply_event(world, "border_raids", {"CIV1": "nomads"}, year=1)
    world, _ = apply_event(world, "declare_war_and_ally", {"CIV1": "nomads", "CIV2": "rival"}, year=2)
    world, record = apply_event(world, "border_raids", {"CIV1": "nomads"}, year=3)
    assert world.relationship("home", "nomads").attitude("home", "nomads") == Attitude.HOSTILITY
    assert not [c for c in record.world_changes if c.field.startswith("attitude")]


def test_slot_conditions_enforced():
    with pytest.raises(EventError, match="conditions"):
        apply_event(make_world(), "border_raids", {"CIV1": "rival"}, year=0)
    template = load_event_catalog().event("border_raids")
    assert [c.id for c in slot_candidates(make_world(), template, "CIV1")] == ["nomads"]


@pytest.mark.parametrize("bindings, message", [
    ({"CIV1": "rival"}, "needs slots"),
    ({"CIV1": "rival", "CIV2": "rival"}, "different civilization"),
    ({"CIV1": "rival", "CIV2": "atlantis"}, "unknown civilization"),
    ({"CIV1": "rival", "CIV2": "nomads", "CIV3": "home"}, "has no slots"),
    ({"CIV1": "rival", "CIV2": "nomads", "HOME_CIV": "rival"}, "focal civilization"),
])
def test_invalid_bindings(bindings, message):
    with pytest.raises(EventError, match=message):
        apply_event(make_world(), "declare_war_and_ally", bindings, year=0)


def test_unknown_event():
    with pytest.raises(EventError, match="unknown event"):
        apply_event(make_world(), "alien_invasion", {}, year=0)


@pytest.mark.parametrize("effects", [
    {"world_effects": [{"id": "feud", "kind": "set_attitude", "viewer": "CIV1", "target": "CIV2",
                        "attitude": "hostility", "description": "x"}]},
    {"linguistic_effects": [{"id": "feud_loans", "kind": "borrow_vocabulary", "donor": "CIV1",
                             "recipient": "CIV2", "semantic_categories": ["conflict"], "description": "x"}]},
])
def test_effects_between_neighbors_rejected(effects):
    with pytest.raises(ValidationError, match="must involve HOME_CIV"):
        EventCatalog.model_validate(effects)


def test_zero_step_shift_rejected():
    with pytest.raises(ValidationError, match="non-zero"):
        EventCatalog.model_validate({"world_effects": [
            {"id": "noop", "kind": "shift_attitude", "viewer": "HOME_CIV", "target": "CIV1", "steps": 0,
             "description": "x"}]})


@pytest.mark.parametrize("event, message", [
    ({"id": "e", "name": "E", "description_template": "[HOME_CIV] meets [CIV9]"}, "undeclared placeholders"),
    ({"id": "e", "name": "E", "description_template": "x", "world_effect_ids": ["nope"]}, "unknown world effect"),
    ({"id": "e", "name": "E", "description_template": "x", "world_effect_ids": ["hostile"]}, "does not declare"),
    ({"id": "e", "name": "E", "description_template": "x", "slots": [{"name": "ENEMY"}]}, "pattern"),
])
def test_catalog_validation(event, message):
    effect = {"id": "hostile", "kind": "set_attitude", "viewer": "HOME_CIV", "target": "CIV1",
              "attitude": "hostility", "description": "x"}
    with pytest.raises(ValidationError, match=message):
        EventCatalog.model_validate({"world_effects": [effect], "events": [event]})
