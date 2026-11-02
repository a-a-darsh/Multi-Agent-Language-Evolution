from uuid import uuid4

from pydantic import BaseModel, ConfigDict


def new_id() -> str:
    return str(uuid4())


class DomainModel(BaseModel):
    """Base for all domain models.

    `extra="forbid"` makes unexpected fields an error instead of silently
    dropping them - important once LLM output is parsed into these models.
    """

    model_config = ConfigDict(extra="forbid")
