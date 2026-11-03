"""Procedural generation of the focal civilization's missing traits and its neighbors.

All structural facts (geography, economy, traits, power, attitudes, contact)
come from seeded weighted tables in `data/world_generation.json`. Narrative
descriptions are left for the LLM layer, which must stay consistent with them.
"""

import json
import random
from collections.abc import Mapping, Sequence
from functools import cache

from pydantic import Field

from app.data import DATA_DIR
from app.generation.phoneme_generator import derive_rng
from app.generation.root_generator import generate_root
from app.models.civilization import (
    Attitude,
    Civilization,
    CivilizationRelationship,
    ContactIntensity,
    ContactType,
    LanguageSketch,
    RelativePower,
    Separation,
    World,
)
from app.models.common import DomainModel
from app.models.eras import ERA_DEFAULT_YEAR, era_for_year
from app.models.generation import CivilizationProfile, GenerationRequest
from app.models.phonology import Phoneme, Phonology

# When the user leaves the era to the AI: ancient eras are the natural start for a long history.
ERA_WEIGHTS = {"stone_age": 1, "bronze_age": 3, "iron_age": 3, "classical": 2, "medieval": 1, "early_modern": 0.5}

Weights = dict[str, float]


class Range(DomainModel):
    min: int
    max: int


class TraitConfig(DomainModel):
    options: list[str]
    count: Range
    base_weight: float
    bonus_by_economy: dict[str, Weights]
    incompatible: list[tuple[str, str]]


class AttitudeModifier(DomainModel):
    # Keys are "viewer.<field>", "target.<field>", or "target.power";
    # all conditions must match (any listed value counts as a match).
    when: dict[str, list[str]]
    add: dict[Attitude, float]


class AttitudeConfig(DomainModel):
    base: dict[Attitude, float]
    modifiers: list[AttitudeModifier] = Field(default_factory=list)


class WorldGenerationConfig(DomainModel):
    neighbor_count: Range
    neighbor_geography: dict[str, Weights]
    economy_by_geography: dict[str, Weights]
    social_organization_by_economy: dict[str, Weights]
    traits: TraitConfig
    relative_power: dict[RelativePower, float]
    attitude: AttitudeConfig
    contact_intensity_by_separation: dict[Separation, dict[ContactIntensity, float]]


@cache
def load_world_config() -> WorldGenerationConfig:
    raw = json.loads((DATA_DIR / "world_generation.json").read_text(encoding="utf-8-sig"))
    return WorldGenerationConfig.model_validate(raw)


def _weighted(rng: random.Random, weights: Mapping) -> str:
    options = [k for k, w in weights.items() if w > 0]
    return rng.choices(options, weights=[weights[k] for k in options])[0]


# --- civilization traits -----------------------------------------------------

def _sample_traits(rng: random.Random, economy: str, config: WorldGenerationConfig) -> list[str]:
    weights = {t: config.traits.base_weight + config.traits.bonus_by_economy.get(economy, {}).get(t, 0)
               for t in config.traits.options}
    chosen: list[str] = []
    for _ in range(rng.randint(config.traits.count.min, config.traits.count.max)):
        allowed = {
            t: w for t, w in weights.items()
            if t not in chosen
            and not any({t, c} == set(pair) for c in chosen for pair in config.traits.incompatible)
        }
        if not allowed:
            break
        chosen.append(_weighted(rng, allowed))
    return chosen


_NAME_CONSONANTS = "ptkbdgmnslrvfhzy"


def generate_sound_profile(rng: random.Random) -> Phonology:
    """A small random sound system, distinct enough that each neighbor sounds different."""
    consonants = rng.sample(_NAME_CONSONANTS, rng.randint(6, 10))
    vowels = rng.sample("aeiou", rng.randint(3, 5))
    return Phonology(
        phonemes=[Phoneme.from_grapheme(g) for g in consonants + vowels],
        syllable_patterns=rng.choice([["CV"], ["CV", "CVC"], ["V", "CV", "CVC"]]),
    )


def generate_name(rng: random.Random, phonology: Phonology | None = None) -> str:
    """A name that obeys `phonology`, or a throwaway sound system if none is given."""
    phonology = phonology or generate_sound_profile(rng)
    return phonology.spell(generate_root(phonology, rng.choice([2, 2, 3]), rng)).capitalize()


def _unique_name(rng: random.Random, phonology: Phonology, taken: set[str]) -> str:
    while (name := generate_name(rng, phonology)) in taken:
        pass
    return name


def complete_civilization(
    profile: CivilizationProfile,
    rng: random.Random,
    founding_year: int | None = None,
    config: WorldGenerationConfig | None = None,
) -> Civilization:
    """Fill every field the user left empty; user-given values are kept as is."""
    config = config or load_world_config()
    generated: list[str] = []
    if founding_year is not None:  # explicit override (e.g. scripts and tests)
        profile = profile.model_copy(update={"founding_year": founding_year})

    def fill(field: str, value_if_missing):
        current = getattr(profile, field)
        if current not in (None, []):
            return current
        generated.append(field)
        return value_if_missing()

    geography = fill("geography", lambda: rng.choice(sorted(config.neighbor_geography)))
    economy = fill("economy", lambda: _weighted(rng, config.economy_by_geography[geography]))
    social = fill("social_organization",
                  lambda: _weighted(rng, config.social_organization_by_economy[economy]))
    traits = fill("cultural_traits", lambda: _sample_traits(rng, economy, config))
    name = fill("name", lambda: generate_name(rng))
    # Era and year: a given year implies its era; a given era implies a typical year.
    # Drawn last so adding eras didn't change the other fields for existing seeds.
    year = profile.founding_year
    era = fill("era", lambda: era_for_year(year) if year is not None else _weighted(rng, ERA_WEIGHTS))
    if year is None:
        generated.append("founding_year")
        year = ERA_DEFAULT_YEAR[era]
    return Civilization(name=name, founding_year=year, era=era, geography=geography, economy=economy,
                        social_organization=social, cultural_traits=traits,
                        description=profile.description, generated_fields=generated)


def generate_neighbor(
    rng: random.Random, focal: Civilization, taken_names: set[str], config: WorldGenerationConfig
) -> tuple[Civilization, LanguageSketch]:
    """A neighbor plus a sketch of its language: a sound system and no words yet."""
    phonology = generate_sound_profile(rng)
    geography = _weighted(rng, config.neighbor_geography[focal.geography])
    economy = _weighted(rng, config.economy_by_geography[geography])
    neighbor = Civilization(
        name=_unique_name(rng, phonology, taken_names),
        founding_year=focal.founding_year,
        era=focal.era,  # neighbors live in the same time
        geography=geography,
        economy=economy,
        social_organization=_weighted(rng, config.social_organization_by_economy[economy]),
        cultural_traits=_sample_traits(rng, economy, config),
        generated_fields=["name", "geography", "economy", "social_organization", "cultural_traits"],
    )
    return neighbor, LanguageSketch(civilization_id=neighbor.id, phonology=phonology)


# --- relationships -----------------------------------------------------------

def _condition_matches(key: str, values: Sequence[str], viewer: Civilization, target: Civilization,
                       target_power: RelativePower) -> bool:
    side, field = key.split(".")
    if field == "power":
        actual: object = target_power.value
    else:
        actual = getattr(viewer if side == "viewer" else target, field)
    if isinstance(actual, list):
        return bool(set(actual) & set(values))
    return actual in values


def attitude_weights(viewer: Civilization, target: Civilization, target_power: RelativePower,
                     config: WorldGenerationConfig) -> dict[Attitude, float]:
    """How likely each attitude is for `viewer` toward `target`. Exposed for tests and tuning."""
    weights = dict(config.attitude.base)
    for modifier in config.attitude.modifiers:
        if all(_condition_matches(k, v, viewer, target, target_power) for k, v in modifier.when.items()):
            for attitude, delta in modifier.add.items():
                weights[attitude] = weights.get(attitude, 0) + delta
    # Negative modifiers can reduce an attitude but never rule it out.
    return {a: max(w, 0.05) for a, w in weights.items()}


def _separation(rng: random.Random, a: Civilization, b: Civilization) -> Separation:
    geographies = {a.geography, b.geography}
    if "island" in geographies:
        return Separation.SEA
    weights = {Separation.LAND_BORDER: 4.0}
    if geographies == {"coastal"}:
        weights[Separation.SEA] = 2
    if "river_valley" in geographies:
        weights[Separation.RIVER] = 4 if geographies == {"river_valley"} else 2
    if "mountain" in geographies:
        weights[Separation.MOUNTAINS] = 3
    if "desert" in geographies:
        weights[Separation.DESERT] = 3
    return Separation(_weighted(rng, weights))


def _contact_types(rng: random.Random, a: Civilization, b: Civilization, attitudes: set[Attitude],
                   power: RelativePower, intensity: ContactIntensity) -> list[ContactType]:
    economies = {a.economy, b.economy}
    traits = set(a.cultural_traits) | set(b.cultural_traits)
    hostile = Attitude.HOSTILITY in attitudes
    commercial = bool(economies & {"mercantile", "seafaring"}) or "commercial" in traits
    chances = {
        ContactType.TRADE: 0.4 + 0.4 * commercial - 0.3 * hostile,
        ContactType.WARFARE: (0.7 if hostile else 0.25 if Attitude.DISTRUST in attitudes else 0.05)
                             + 0.15 * bool(traits & {"militaristic", "expansionist"}),
        ContactType.RAIDING: 0.5 if "pastoral_nomadic" in economies
                                    and economies & {"agricultural", "mercantile"} else 0.05,
        ContactType.TRIBUTE: 0.5 if power in (RelativePower.MUCH_WEAKER, RelativePower.MUCH_STRONGER) else 0.05,
        ContactType.DIPLOMACY: 0.1 if hostile else 0.3,
        ContactType.RELIGIOUS: 0.4 if "religious" in traits else 0.1,
        ContactType.INTERMARRIAGE: 0.4 if intensity == ContactIntensity.INTENSE and not hostile else 0.05,
        ContactType.MIGRATION: 0.1,
    }
    chosen = [t for t, p in chances.items() if rng.random() < p]
    return chosen or [ContactType.WARFARE if hostile else ContactType.TRADE]


def generate_relationship(rng: random.Random, focal: Civilization, neighbor: Civilization,
                          config: WorldGenerationConfig) -> CivilizationRelationship:
    neighbor_power = RelativePower(_weighted(rng, config.relative_power))
    focal_to_neighbor = Attitude(_weighted(rng, attitude_weights(focal, neighbor, neighbor_power, config)))
    neighbor_to_focal = Attitude(_weighted(rng, attitude_weights(neighbor, focal, neighbor_power.inverse(), config)))
    separation = _separation(rng, focal, neighbor)
    intensity = ContactIntensity(_weighted(rng, config.contact_intensity_by_separation[separation]))
    return CivilizationRelationship(
        civilization_a_id=focal.id,
        civilization_b_id=neighbor.id,
        attitude_a_to_b=focal_to_neighbor,
        attitude_b_to_a=neighbor_to_focal,
        power_of_b_relative_to_a=neighbor_power,
        separation=separation,
        contact_intensity=intensity,
        contact_types=_contact_types(rng, focal, neighbor, {focal_to_neighbor, neighbor_to_focal},
                                     neighbor_power, intensity),
    )


# --- entry point -------------------------------------------------------------

def generate_world(
    request: GenerationRequest,
    founding_year: int | None = None,
    neighbor_count: int | None = None,
    config: WorldGenerationConfig | None = None,
) -> World:
    config = config or load_world_config()
    focal = complete_civilization(request.civilization, derive_rng(request.seed, "world", "focal"),
                                  founding_year, config)
    rng = derive_rng(request.seed, "world", "neighbors")
    count = neighbor_count or rng.randint(config.neighbor_count.min, config.neighbor_count.max)

    neighbors: list[Civilization] = []
    sketches: list[LanguageSketch] = []
    relationships: list[CivilizationRelationship] = []
    taken = {focal.name}
    for _ in range(count):
        neighbor, sketch = generate_neighbor(rng, focal, taken, config)
        taken.add(neighbor.name)
        neighbors.append(neighbor)
        sketches.append(sketch)
        relationships.append(generate_relationship(rng, focal, neighbor, config))

    return World(seed=request.seed, focal_civilization_id=focal.id,
                 civilizations=[focal, *neighbors], relationships=relationships,
                 language_sketches=sketches)
