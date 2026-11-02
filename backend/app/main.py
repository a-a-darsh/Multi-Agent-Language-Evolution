from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import languages, questionnaire, settings, worlds
from app.api.deps import get_generation_service
from app.api.errors import ApiError, api_error_handler


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # Stop the generation worker thread on shutdown. Without this, auto-reload hung
    # ("Reloading..." with no new process) after a generation had run, because the
    # interpreter waits for executor threads before exiting.
    if get_generation_service.cache_info().currsize:
        get_generation_service().executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Language Evolution Simulator", lifespan=lifespan)
app.add_exception_handler(ApiError, api_error_handler)
for module in (questionnaire, worlds, settings, languages):
    app.include_router(module.router, prefix="/api")
