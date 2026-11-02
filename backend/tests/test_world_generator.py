import random
from collections import Counter

import pytest
from pydantic import ValidationError

from app.data import load_questions
from app.generation.phoneme_generator import parse_spelling
from app.generation.world_generator import (
    attitude_weights,
    complete_civilization,
    generate_world,
    load_world_config,
)
from app.models.civilization import (
    Attitude,
    Civilization,
    CivilizationRelationship,
    ContactIntensity,
    ContactType,
    DonorWord,
    LanguageSketch,
    RelativePower,
    Separation,
)
from app.models.generation import CivilizationProfile, GenerationRequest
from app.validation.phonology_validator import validate_phoneme_sequence
from tests.factories import sample_phonology

CONFIG = load_world_config()


def world(seed=42, **civilization):
    return generate_world(GenerationRequest(civilization=CivilizationProfile(**civilization), seed=seed))


def civ(**fields):
    defaults = dict(name="X", founding_year=0, geography="coastal", economy="mixed")
    return Civilization(**{**defaults, **fields})


def content(w):
    """World contents minus random UUIDs, for reproducibility checks."""
    return [(c.name, c.geography, c.economy, c.social_organization, c.cultural_traits) for c in w.civilizations] + [
        (r.attitude_a_to_b, r.attitude_b_to_a, r.power_of_b_relative_to_a, r.separation,
         r.contact_intensity, r.contact_types) for r in w.relationships
    ]


def test_config_values_match_questionnaire_options():
    options = {q.id: {o.value for o in q.options} for q in load_questions()}
    assert set(CONFIG.neighbor_geography) == options["geography"]
    for targets in CONFIG.neighbor_geography.values():
        assert set(targets) <= options["geography"]
    assert set(CONFIG.economy_by_geography) == options["geography"]
    assert set(CONFIG.social_organization_by_economy) == options["economy"]
    for weights in CONFIG.social_organization_by_economy.values():
        assert set(weights) <= options["social_organization"]
    assert set(CONFIG.traits.options) == options["cultural_traits"]
    assert set(CONFIG.contact_intensity_by_separation) == set(Separation)


def test_world_is_reproducible_from_seed():
    assert content(world(seed=7)) == content(world(seed=7))
    assert content(world(seed=7)) != content(world(seed=8))


def test_focal_civilization_independent_of_neighbors():
    # Initial language generation reads only the focal civilization, so it
    # must not change with how many (or which) neighbors are generated.
    request = GenerationRequest(seed=11)
    focal = [generate_world(request, neighbor_count=n).focal for n in (1, 3, 5)]
    assert len({(c.name, c.geography, c.economy, c.social_organization, tuple(c.cultural_traits))
                for c in focal}) == 1


def test_world_shape():
    for seed in range(30):
        w = world(seed=seed, geography="steppe")
        assert CONFIG.neighbor_count.min <= len(w.neighbors) <= CONFIG.neighbor_count.max
        assert len({c.name for c in w.civilizations}) == len(w.civilizations)
        for n in w.neighbors:
            assert n.geography in CONFIG.neighbor_geography["steppe"]
            r = w.relationship(w.focal_civilization_id, n.id)
            assert r.civilization_a_id == w.focal_civilization_id
            assert r.contact_types


def test_user_civilization_fields_preserved_and_rest_marked_generated():
    w = world(name="Savari", geography="coastal", cultural_traits=["religious"])
    focal = w.focal
    assert (focal.name, focal.geography, focal.cultural_traits) == ("Savari", "coastal", ["religious"])
    assert focal.generated_fields == ["economy", "social_organization"]


def test_complete_civilization_follows_geography_tables():
    for seed in range(50):
        c = complete_civilization(CivilizationProfile(geography="steppe"), random.Random(seed))
        assert c.economy in CONFIG.economy_by_geography["steppe"]


def test_incompatible_traits_never_combined():
    for seed in range(100):
        for c in world(seed=seed).civilizations:
            for pair in CONFIG.traits.incompatible:
                assert not set(pair) <= set(c.cultural_traits)


def test_islands_are_separated_by_sea():
    for seed in range(30):
        w = world(seed=seed, geography="island")
        assert all(r.separation == Separation.SEA for r in w.relationships)


def test_attitudes_can_be_one_sided():
    asymmetric = [
        r for seed in range(50) for r in world(seed=seed).relationships
        if Attitude.ADMIRATION in (r.attitude_a_to_b, r.attitude_b_to_a)
        and r.attitude_a_to_b != r.attitude_b_to_a
    ]
    assert asymmetric


def test_mercantile_neighbors_are_admired_more():
    viewer = civ(economy="agricultural")
    trader = civ(economy="mercantile", cultural_traits=["commercial"])
    farmer = civ(economy="agricultural")

    def admiration_share(target):
        w = attitude_weights(viewer, target, RelativePower.EQUAL, CONFIG)
        return w[Attitude.ADMIRATION] / sum(w.values())

    assert admiration_share(trader) > 2 * admiration_share(farmer)


def test_isolationists_distrust_everyone_more():
    target = civ()
    normal = attitude_weights(civ(), target, RelativePower.EQUAL, CONFIG)
    isolationist = attitude_weights(civ(cultural_traits=["isolationist"]), target, RelativePower.EQUAL, CONFIG)
    assert isolationist[Attitude.DISTRUST] > normal[Attitude.DISTRUST]
    assert all(w > 0 for w in isolationist.values())


def test_hostility_tends_to_bring_conflict():
    counts = Counter()
    for seed in range(200):
        for r in world(seed=seed).relationships:
            if Attitude.HOSTILITY in (r.attitude_a_to_b, r.attitude_b_to_a):
                counts["hostile"] += 1
                counts["conflict"] += bool({ContactType.WARFARE, ContactType.RAIDING} & set(r.contact_types))
    assert counts["conflict"] / counts["hostile"] > 0.6


def test_relationship_perspective_helpers():
    r = CivilizationRelationship(
        civilization_a_id="a", civilization_b_id="b",
        attitude_a_to_b=Attitude.ADMIRATION, attitude_b_to_a=Attitude.INDIFFERENCE,
        power_of_b_relative_to_a=RelativePower.STRONGER, separation=Separation.LAND_BORDER,
        contact_intensity=ContactIntensity.REGULAR, contact_types=[ContactType.TRADE],
    )
    assert r.attitude("a", "b") == Attitude.ADMIRATION
    assert r.attitude("b", "a") == Attitude.INDIFFERENCE
    assert r.power_of("b", "a") == RelativePower.STRONGER
    assert r.power_of("a", "b") == RelativePower.WEAKER
    assert r.other("a") == "b"


def test_self_relationship_rejected():
    with pytest.raises(ValidationError):
        CivilizationRelationship(
            civilization_a_id="a", civilization_b_id="a",
            attitude_a_to_b=Attitude.RESPECT, attitude_b_to_a=Attitude.RESPECT,
            power_of_b_relative_to_a=RelativePower.EQUAL, separation=Separation.SEA,
            contact_intensity=ContactIntensity.SPORADIC, contact_types=[ContactType.TRADE],
        )


def test_neighbors_get_empty_language_sketches_with_matching_names():
    w = world(seed=3)
    assert len(w.language_sketches) == len(w.neighbors)
    for n in w.neighbors:
        sketch = w.language_sketch(n.id)
        assert sketch.words == []  # words are created lazily when borrowed
        # The neighbor's name obeys its own sound system.
        assert validate_phoneme_sequence(parse_spelling(n.name, sketch.phonology), sketch.phonology).ok


def test_donor_words_can_build_on_earlier_words():
    horse = DonorWord(concept_id="horse", phonemes=list("pivka"), orthography="pivka")
    rider = DonorWord(concept_id="rider", phonemes=list("pivkar"), orthography="pivkar", base_word_id=horse.id)
    sketch = LanguageSketch(civilization_id="n", phonology=sample_phonology(), words=[horse, rider])
    assert sketch.word_for("rider").base_word_id == sketch.word_for("horse").id
    assert sketch.word_for("cattle") is None
    with pytest.raises(ValidationError, match="unknown word"):
        LanguageSketch(civilization_id="n", phonology=sample_phonology(), words=[rider])
