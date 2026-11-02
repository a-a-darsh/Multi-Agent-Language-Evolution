from enum import StrEnum

from pydantic import Field, model_validator

from app.models.common import DomainModel


class InputType(StrEnum):
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    FREE_TEXT = "free_text"
    SHORT_TEXT = "short_text"  # single line, e.g. a name
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
    # Shown in every session, outside the random selection (e.g. the civilization's name).
    always_ask: bool = False
    # Selects offer "describe your own": free text instead of (or as well as) the options.
    allow_custom: bool = True
    # Dotted path into GenerationRequest where the answer is stored,
    # e.g. "civilization.geography". Keeps the question bank config-driven.
    maps_to: str

    @model_validator(mode="after")
    def _selects_need_options(self) -> "Question":
        is_select = self.input_type in (InputType.SINGLE_SELECT, InputType.MULTI_SELECT)
        if is_select and not self.options:
            raise ValueError(f"question {self.id!r} is a select but has no options")
        return self
