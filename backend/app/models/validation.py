from enum import StrEnum
from typing import Any

from pydantic import Field

from app.models.common import DomainModel


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(DomainModel):
    """A precise, machine-readable problem report.

    These are fed back to the LLM during repair loops, so `code` and
    `details` should be specific enough to act on.
    """

    code: str
    message: str
    object_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    severity: Severity = Severity.ERROR


class ValidationResult(DomainModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, code: str, message: str, **kwargs: Any) -> None:
        self.issues.append(ValidationIssue(code=code, message=message, **kwargs))

    def extend(self, other: "ValidationResult") -> None:
        self.issues.extend(other.issues)
