from fastapi import Request
from fastapi.responses import JSONResponse

from app.models.validation import ValidationIssue


class ApiError(Exception):
    """Consistent error body: {"error": {"code", "message", "issues"}}."""

    def __init__(self, status_code: int, code: str, message: str, issues: list[ValidationIssue] = ()):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.issues = list(issues)


async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message,
                           "issues": [i.model_dump(mode="json") for i in exc.issues]}},
    )
