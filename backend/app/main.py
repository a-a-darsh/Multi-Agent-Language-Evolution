from fastapi import FastAPI

from app.api import questionnaire, settings, worlds
from app.api.errors import ApiError, api_error_handler

app = FastAPI(title="Language Evolution Simulator")
app.add_exception_handler(ApiError, api_error_handler)
app.include_router(questionnaire.router, prefix="/api")
app.include_router(worlds.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
