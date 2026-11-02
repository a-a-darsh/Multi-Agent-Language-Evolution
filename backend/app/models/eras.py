"""Technological eras. The world is fictional, so an era describes a technology level;
real-world dates are rough anchors for turning a user's year into an era and back."""

import re

ERAS: tuple[str, ...] = ("stone_age", "bronze_age", "iron_age", "classical", "medieval", "early_modern")

# First year of each era (negative = BCE), used to classify a user's year.
ERA_START: dict[str, int] = {
    "stone_age": -100_000,
    "bronze_age": -3300,
    "iron_age": -1200,
    "classical": -500,
    "medieval": 500,
    "early_modern": 1500,
}

# Founding year used when the user picks an era without a year.
ERA_DEFAULT_YEAR: dict[str, int] = {
    "stone_age": -5000,
    "bronze_age": -2000,
    "iron_age": -800,
    "classical": 0,
    "medieval": 1000,
    "early_modern": 1600,
}


def era_index(era: str) -> int:
    return ERAS.index(era)


def era_for_year(year: int) -> str:
    return next(era for era in reversed(ERAS) if year >= ERA_START[era])


def parse_year(text: str) -> int | None:
    """'around 1500 BCE' -> -1500; 'year 300' -> 300. None if no year is mentioned."""
    match = re.search(r"(\d[\d,]*)\s*(bce|bc|b\.c\.e?\.?|ce|ad|a\.d\.)?", text.lower())
    if not match:
        return None
    year = int(match.group(1).replace(",", ""))
    era = (match.group(2) or "").replace(".", "")
    return -year if era in ("bce", "bc") else year


def format_year(year: int) -> str:
    return f"{-year} BCE" if year < 0 else f"{year} CE"
