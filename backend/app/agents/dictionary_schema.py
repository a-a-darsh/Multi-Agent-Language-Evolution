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

    @property
    def bases(self) -> list[str]:
        return [self.base_concept_id]

    def describe(self) -> str:
        return f"{self.base_concept_id} + {self.rule_id}"


class CompoundEntry(DomainModel):
    """One planned compound: concept = first + second (two existing words joined)."""

    concept_id: str
    first_concept_id: str
    second_concept_id: str
    explanation: str = ""

    @property
    def bases(self) -> list[str]:
        return [self.first_concept_id, self.second_concept_id]

    def describe(self) -> str:
        return f"{self.first_concept_id} + {self.second_concept_id} (compound)"


Planned = PlanEntry | CompoundEntry


class SenseExtension(DomainModel):
    """Polysemy: `concept_id` is expressed by the same word as `base_concept_id`
    (e.g. wealth by the word for zinc, where zinc is currency)."""

    concept_id: str
    base_concept_id: str
    explanation: str = ""


def _literal(values: list[str]):
    return Literal[tuple(values)]


def plan_schema(derivable_ids: list[str], concept_ids: list[str], rule_ids: list[str],
                max_derivations: int = MAX_PLANNED_DERIVATIONS,
                compoundable_ids: list[str] | None = None) -> type[BaseModel]:
    """`compoundable_ids` (default: all derivable) are the concepts that may be compounds."""
    compoundable_ids = compoundable_ids if compoundable_ids is not None else derivable_ids
    derivation = create_model(
        "PlannedDerivation",
        concept_id=(_literal(derivable_ids), Field(description="The concept that will be a derived word.")),
        base_concept_id=(_literal(concept_ids), Field(description=(
            "The existing word whose meaning it is built from - choose for meaning first."))),
        rule_id=(_literal(rule_ids), Field(description="The affix that turns the base's meaning into the concept's.")),
        explanation=(str, Field(max_length=80, description="Show the composition, e.g. 'sea + agent -> sailor'.")),
    )
    compound = create_model(
        "PlannedCompound",
        concept_id=(_literal(compoundable_ids), Field(description="The concept that will be a compound word.")),
        first_concept_id=(_literal(concept_ids), Field(description="The first word of the compound.")),
        second_concept_id=(_literal(concept_ids), Field(description="The second word of the compound.")),
        explanation=(str, Field(max_length=80, description="Show the composition, e.g. 'sea + horse -> seahorse'.")),
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
        compounds=(list[compound], Field(max_length=max_derivations, description=(
            "Concepts built by joining two other concepts' words, when the concept really combines both "
            "meanings and no affix says it. Not for basic vocabulary."))),
        derivations=(list[derivation], Field(max_length=max_derivations, description=(
            "Concepts built from ONE other concept with an affix, only where base + affix says everything "
            "important about the concept. Every concept not listed anywhere gets its own root."))),
        extended_senses=(list[extension], Field(max_length=MAX_EXTENDED_SENSES, description=(
            "A few meanings expressed by the SAME word as another concept, because this culture links them "
            "(e.g. where zinc is money, one word for zinc and wealth). Use sparingly; may be empty."))),
    )


def repair_schema(failing_ids: list[str], concept_ids: list[str], rule_ids: list[str]) -> type[BaseModel]:
    """Only the failing concepts may appear: the model can't touch anything else."""
    fix = create_model(
        "PlanFix",
        concept_id=(_literal(failing_ids), Field(description="A concept whose derivation had a problem.")),
        action=(Literal["derive", "compound", "extend", "independent_root"], Field(description=(
            "'derive' with a base and rule; 'compound' joining base and second_base; 'extend' to share the "
            "base's word as an extra meaning; or 'independent_root' to give it its own unrelated word."))),
        base_concept_id=(_literal(concept_ids) | None, Field(description=(
            "Base for 'derive' or 'extend', first word for 'compound', else null."))),
        second_base_concept_id=(_literal(concept_ids) | None, Field(description=(
            "Second word for 'compound', else null."))),
        rule_id=(_literal(rule_ids) | None, Field(description="Rule for 'derive', else null.")),
    )
    # Bounded: unbounded, the model once repeated the same fix 121 times until the reply broke.
    return create_model("PlanRepair", fixes=(list[fix], Field(max_length=len(failing_ids),
                                                              description="Exactly one fix per listed concept.")))


def review_schema(derived_ids: list[str]) -> type[BaseModel]:
    """Critic: one judgment per derivation or shared word, showing its work before deciding."""
    review = create_model(
        "DerivationReview",
        concept_id=(_literal(derived_ids), ...),
        literal_meaning=(str, Field(max_length=100, description=(
            "What the parts literally say ('person associated with the sea', 'sea + horse'), or for a shared "
            "word, how the two meanings connect ('zinc is money, so zinc means wealth')."))),
        missing=(str, Field(max_length=80, description=(
            "An important quality of the concept that the parts do NOT say (for oar = swim + noun: 'a tool; "
            "rowing, not swimming'), or 'none' if nothing important is missing."))),
        plausible=(bool, Field(description=(
            "true if speakers could naturally use that for the concept (like 'sea-person' for sailor) and "
            "nothing important is missing. false if the connection is forced, wrong, or too vague."))),
    )
    return create_model("PlanReview", reviews=(list[review], Field(description="One review per derivation.")))
