"""Schemas for the morphological planner, built at run time.

Each run has its own concepts and rules, so schemas are created dynamically
with `create_model`, and ids become Literal enums the model must choose from.

Lessons learned (see docs/what-went-wrong.md):
- A flat (concept, base, rule) list gave good *semantic* pairings but many
  part-of-speech errors.
- One list per rule made POS errors impossible but wrecked the semantics: the
  model picked the rule first and then mechanically walked down the verb list
  ("hair = eat + agent", "harbor = walk + agent").
So the schema follows the natural order of the decision (concept, then base,
then rule), and part-of-speech fit is handled by code in `check_plan`, which
silently corrects the rule when exactly one rule fits.
"""

from typing import Literal

from pydantic import BaseModel, Field, create_model

from app.models.common import DomainModel

MAX_PLANNED_DERIVATIONS = 70


MAX_EXTENDED_SENSES = 12


class PlanEntry(DomainModel):
    """One planned derivation: concept = base + rule."""

    concept_id: str
    base_concept_id: str
    rule_id: str
    explanation: str = ""


class SenseExtension(DomainModel):
    """Polysemy: `concept_id` is expressed by the same word as `base_concept_id`
    (e.g. wealth by the word for zinc, where zinc is currency)."""

    concept_id: str
    base_concept_id: str
    explanation: str = ""


def _literal(values: list[str]):
    return Literal[tuple(values)]


def plan_schema(derivable_ids: list[str], concept_ids: list[str], rule_ids: list[str],
                max_derivations: int = MAX_PLANNED_DERIVATIONS) -> type[BaseModel]:
    derivation = create_model(
        "PlannedDerivation",
        concept_id=(_literal(derivable_ids), Field(description="The concept that will be a derived word.")),
        base_concept_id=(_literal(concept_ids), Field(description=(
            "The existing word whose meaning it is built from - choose for meaning first."))),
        rule_id=(_literal(rule_ids), Field(description="The affix that turns the base's meaning into the concept's.")),
        explanation=(str, Field(max_length=80, description="Show the composition, e.g. 'sea + agent -> sailor'.")),
    )
    extension = create_model(
        "ExtendedSense",
        concept_id=(_literal(derivable_ids), Field(description="The meaning that will share another word.")),
        base_concept_id=(_literal(concept_ids), Field(description="The concept whose word it shares.")),
        explanation=(str, Field(max_length=80, description="Why one word covers both, e.g. 'zinc is their currency'.")),
    )
    return create_model(
        "MorphologicalPlan",
        reasoning=(str, Field(max_length=1200, description=(
            "Briefly: which semantic families matter most for this culture, and how you'll use the affixes."))),
        derivations=(list[derivation], Field(max_length=max_derivations, description=(
            "Concepts built from another concept with an affix. Every concept not listed gets its own root."))),
        extended_senses=(list[extension], Field(max_length=MAX_EXTENDED_SENSES, description=(
            "A few meanings expressed by the SAME word as another concept, because this culture links them "
            "(e.g. where zinc is money, one word for zinc and wealth). Use sparingly; may be empty."))),
    )


def repair_schema(failing_ids: list[str], concept_ids: list[str], rule_ids: list[str]) -> type[BaseModel]:
    """Only the failing concepts may appear: the model can't touch anything else."""
    fix = create_model(
        "PlanFix",
        concept_id=(_literal(failing_ids), Field(description="A concept whose derivation had a problem.")),
        action=(Literal["derive", "extend", "independent_root"], Field(description=(
            "'derive' with a base and rule; 'extend' to share the base's word as an extra meaning; "
            "or 'independent_root' to give it its own unrelated word."))),
        base_concept_id=(_literal(concept_ids) | None, Field(description="Base for 'derive' or 'extend', else null.")),
        rule_id=(_literal(rule_ids) | None, Field(description="Rule for 'derive', else null.")),
    )
    return create_model("PlanRepair", fixes=(list[fix], Field(description="Exactly one fix per listed concept.")))


def review_schema(derived_ids: list[str]) -> type[BaseModel]:
    """Critic: one judgment per derivation or shared word, showing its work before deciding."""
    review = create_model(
        "DerivationReview",
        concept_id=(_literal(derived_ids), ...),
        literal_meaning=(str, Field(max_length=100, description=(
            "What base + affix literally means ('person associated with the sea'), or for a shared word, "
            "how the two meanings connect ('zinc is money, so zinc means wealth')."))),
        plausible=(bool, Field(description=(
            "true if speakers could naturally use that for the concept (like 'sea-person' for sailor). "
            "false if the connection is forced or wrong."))),
    )
    return create_model("PlanReview", reviews=(list[review], Field(description="One review per derivation.")))
