"""Choose which meanings the founding dictionary covers.

Every core concept is included. Cultural concepts are included when their
tags match the civilization (era, geography, economy, society, traits), most
relevant first; the era counts double, since technology decides what people
need words for. Concepts that don't exist yet in the civilization's era
(a Bronze Age people has no coins or gunpowder) are never included.
Culture adds specialized vocabulary; it never removes basic vocabulary.
"""

from collections.abc import Iterable

from app.models.civilization import Civilization
from app.models.eras import era_index
from app.models.lexicon import ConceptPriority, SemanticConcept

MAX_CULTURAL_CONCEPTS = 45
ERA_WEIGHT = 2


def civilization_tags(civ: Civilization) -> set[str]:
    return {t for t in (civ.geography, civ.economy, civ.social_organization, *civ.cultural_traits) if t}


def exists_in_era(concept: SemanticConcept, era: str | None) -> bool:
    if concept.min_era is None or era is None:
        return True
    return era_index(concept.min_era) <= era_index(era)


def select_concepts(inventory: Iterable[SemanticConcept], civ: Civilization,
                    required_ids: Iterable[str] = (), max_cultural: int = MAX_CULTURAL_CONCEPTS) -> list[SemanticConcept]:
    """Core concepts, then the best-matching cultural ones, plus any `required_ids` (e.g. user words)."""
    inventory = list(inventory)
    tags = civilization_tags(civ)
    required = set(required_ids)

    def score(c: SemanticConcept) -> int:
        return len(tags & set(c.cultural_tags)) + (ERA_WEIGHT if civ.era in c.cultural_tags else 0)

    core = [c for c in inventory if c.priority == ConceptPriority.CORE]
    scored = [(score(c), c) for c in inventory
              if c.priority == ConceptPriority.CULTURAL and exists_in_era(c, civ.era)]
    # Higher score first; inventory order breaks ties, so selection is deterministic.
    matching = [c for s, c in sorted(scored, key=lambda sc: -sc[0]) if s > 0][:max_cultural]

    # User words are always included, even anachronistic ones: the user asked for them.
    chosen = {c.id for c in core} | {c.id for c in matching} | required
    return [c for c in inventory if c.id in chosen]
