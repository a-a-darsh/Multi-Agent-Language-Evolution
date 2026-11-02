from enum import StrEnum

from pydantic import Field, model_validator

from app.models.common import DomainModel


class InputType(StrEnum):
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    FREE_TEXT = "free_text"
    YES_NO = "yes_no"
    # One word for a fixed concept; maps_to ends in the concept id.
    CUSTOM_WORD = "custom_word"
    # Any number of {concept_id: spelling} pairs.
    CUSTOM_VOCABULARY = "custom_vocabulary"


class QuestionOption(DomainModel):
    value: str
    label: str
    description: str | None = None


class Question(DomainModel):
    id: str
    category: str
    prompt: str
    help: str | None = None
    input_type: InputType
    options: list[QuestionOption] = Field(default_factory=list)
    required: bool = False
    # Dotted path into GenerationRequest where the answer is stored,
    # e.g. "civilization.geography". Keeps the question bank config-driven.
    maps_to: str

    @model_validator(mode="after")
    def _selects_need_options(self) -> "Question":
        is_select = self.input_type in (InputType.SINGLE_SELECT, InputType.MULTI_SELECT)
        if is_select and not self.options:
            raise ValueError(f"question {self.id!r} is a select but has no options")
        return self
