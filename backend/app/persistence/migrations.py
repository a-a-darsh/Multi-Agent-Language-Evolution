"""Upgrades for stored JSON written by older versions of the domain models.

Runs on raw dicts before Pydantic validation, so old records load without the
models having to carry legacy code paths.

Phoneme ids: languages stored before the sound catalog used the letter as the
phoneme id ("sh", "y", "c"). Ids are now IPA symbols ("ʃ", "j", "ts"). Every
stored phoneme already recorded its IPA, so the mapping is exact and per
language: an old "c" recorded as /tʃ/ becomes id "tʃ" still spelled "c".
"""

from typing import Any

Json = dict[str, Any]


class MigrationError(ValueError):
    pass


def _phoneme_id_map(phonology: Json) -> dict[str, str]:
    """old id -> IPA id, or {} if the phonology already uses IPA ids."""
    mapping = {p["id"]: p["ipa"] for p in phonology.get("phonemes", [])}
    if all(old == new for old, new in mapping.items()):
        return {}
    if len(set(mapping.values())) != len(mapping):
        raise MigrationError(f"two stored phonemes share one IPA value: {sorted(mapping.items())}")
    return mapping


def _rename(ids: list[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(p, p) for p in ids]


def _upgrade_phonology(phonology: Json, mapping: dict[str, str]) -> None:
    for p in phonology.get("phonemes", []):
        p["id"] = p["ipa"]
    for r in phonology.get("restrictions", []):
        r["phoneme_id"] = mapping.get(r["phoneme_id"], r["phoneme_id"])


def upgrade_language_state(state: Json) -> Json:
    version = state["version"]
    mapping = _phoneme_id_map(version["phonology"])
    if not mapping:
        return state
    _upgrade_phonology(version["phonology"], mapping)
    repair = version.get("boundary_repair") or {}
    for field in ("linking_vowel", "linking_consonant"):
        if repair.get(field):
            repair[field] = mapping.get(repair[field], repair[field])
    for m in state.get("morphemes", []):
        m["phonemes"] = _rename(m["phonemes"], mapping)
    for f in state.get("forms", []):
        f["phonemes"] = _rename(f["phonemes"], mapping)
        for seg in f.get("segments", []):
            seg["phonemes"] = _rename(seg["phonemes"], mapping)
    return state


def upgrade_world(world: Json) -> Json:
    for sketch in world.get("language_sketches", []):
        mapping = _phoneme_id_map(sketch["phonology"])
        if not mapping:
            continue
        _upgrade_phonology(sketch["phonology"], mapping)
        for w in sketch.get("words", []):
            w["phonemes"] = _rename(w["phonemes"], mapping)
    return world
