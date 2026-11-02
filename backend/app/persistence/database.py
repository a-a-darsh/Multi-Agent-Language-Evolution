"""SQLite storage via SQLAlchemy.

Design decision: languages are stored as JSON documents of the Pydantic
domain models, plus a few indexed columns for listing. Domain objects never
depend on the ORM (the plan's requirement), and the repository is the only
code that knows how they are stored. Normalizing into per-entity tables
(lexemes, morphemes...) can come later without touching callers.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / ".local" / "langevo.sqlite3"


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


class GenerationRow(Base):
    """One generation job: its inputs, live progress, and full outcome record."""

    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    status: Mapped[str] = mapped_column(String)  # queued | running | succeeded | failed
    current_step: Mapped[str | None] = mapped_column(String, nullable=True)
    steps_json: Mapped[str] = mapped_column(Text, default="[]")
    request_json: Mapped[str] = mapped_column(Text)
    model_json: Mapped[str] = mapped_column(Text, default="{}")
    outcome_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    language_id: Mapped[str | None] = mapped_column(String, nullable=True)


class LanguageRow(Base):
    __tablename__ = "languages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    name: Mapped[str] = mapped_column(String)
    civilization_name: Mapped[str] = mapped_column(String)
    entry_count: Mapped[int] = mapped_column(Integer)
    derived_count: Mapped[int] = mapped_column(Integer)
    generation_id: Mapped[str] = mapped_column(ForeignKey("generations.id"))
    state_json: Mapped[str] = mapped_column(Text)   # LanguageState
    world_json: Mapped[str] = mapped_column(Text)   # World (civilization + neighbors)


def make_session_factory(path: Path | None = None) -> sessionmaker:
    path = Path(path or os.environ.get("LANGEVO_DB_PATH", DEFAULT_DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
