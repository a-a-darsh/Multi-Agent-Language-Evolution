import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.generation.phoneme_generator import (
    GenerationError,
    UnsupportedSpellingError,
    derive_rng,
    generate_syllable,
    parse_spelling,
    render_phonemes,
)
from app.generation.root_generator import DuplicateFormError, RootRegistry, generate_root
from app.models.phonology import SUPPORTED_PATTERNS, Phoneme, Phonology, Position, PositionRestriction
from app.models.sounds import sound_for_grapheme
from app.validation.phonology_validator import validate_phoneme_sequence
from tests.factories import sample_phonology


@st.composite
def phonologies(draw) -> Phonology:
    """Random but valid phonologies, to check generator/validator agreement broadly."""
    consonants = draw(st.lists(st.sampled_from([*"ptkbdgmnslrvfhwyc", "sh", "ch", "th", "ng", "zh"]),
                               min_size=2, max_size=12, unique=True))
    vowels = draw(st.lists(st.sampled_from("aeiou"), min_size=1, max_size=5, unique=True))
    patterns = draw(st.lists(st.sampled_from(sorted(SUPPORTED_PATTERNS)), min_size=1, max_size=4, unique=True))
    # Forbid one consonant somewhere, keeping at least one consonant free for every slot.
    restricted = draw(st.sampled_from(consonants))
    positions = draw(st.lists(st.sampled_from(list(Position)), min_size=1, max_size=2, unique=True))
    return Phonology(
        phonemes=[Phoneme.from_grapheme(g) for g in consonants + vowels],
        syllable_patterns=patterns,
        restrictions=[PositionRestriction(phoneme_id=sound_for_grapheme(restricted).ipa,
                                          forbidden_positions=positions)],
        allow_hiatus=draw(st.booleans()),
    )


@settings(max_examples=200, deadline=None)
@given(phonology=phonologies(), seed=st.integers(0, 2**31), syllables=st.integers(1, 3))
def test_every_generated_root_passes_validation(phonology, seed, syllables):
    try:
        root = generate_root(phonology, syllables, derive_rng(seed, "test"))
    except GenerationError:
        # Legitimate for degenerate inventories, e.g. only "V" with no hiatus allowed.
        return
    assert validate_phoneme_sequence(root, phonology).ok


def test_syllable_matches_pattern():
    phonology = sample_phonology()
    rng = derive_rng(1, "s")
    for _ in range(100):
        syllable = generate_syllable(phonology, rng, "CVC")
        assert len(syllable) == 3 and syllable[2] != "v"  # /v/ is forbidden in codas


def test_derive_rng_is_reproducible_and_independent():
    assert derive_rng(42, "roots").random() == derive_rng(42, "roots").random()
    assert derive_rng(42, "roots").random() != derive_rng(42, "affixes").random()


def test_spelling_round_trip():
    phonology = sample_phonology()
    assert render_phonemes(parse_spelling("Nalka", phonology), phonology) == "nalka"
    with pytest.raises(UnsupportedSpellingError) as exc:
        parse_spelling("nal-ka", phonology)
    assert exc.value.unsupported == ["-"]


def test_digraphs_are_single_sounds():
    assert parse_spelling("shira") == ["ʃ", "i", "r", "a"]
    assert parse_spelling("ngacha") == ["ŋ", "a", "tʃ", "a"]
    # A language without /ʃ/ reads the same letters as two sounds.
    no_sh = sample_phonology(phonemes=[Phoneme.from_grapheme(g) for g in [*"shira"]], restrictions=[])
    assert parse_spelling("shira", no_sh) == ["s", "h", "i", "r", "a"]
    with_sh = sample_phonology(phonemes=[Phoneme.from_grapheme(g) for g in ["sh", "r", "i", "a"]], restrictions=[])
    assert render_phonemes(["ʃ", "i", "r", "a"], with_sh) == "shira"


def test_registry_generates_unique_reproducible_roots():
    phonology = sample_phonology()
    a = RootRegistry(phonology, seed=42).generate_candidates(150)
    b = RootRegistry(phonology, seed=42).generate_candidates(150)
    assert [c.phonemes for c in a] == [c.phonemes for c in b]
    assert len({tuple(c.phonemes) for c in a}) == 150
    assert [c.provenance.index for c in a] == list(range(150))


def test_registry_never_offers_reserved_forms():
    registry = RootRegistry(sample_phonology(), seed=1)
    candidates = registry.generate_candidates(20)
    registry.reserve(candidates[0].phonemes, owner_id="user:water", owner_kind="user_word")
    assert candidates[0].id not in {c.id for c in registry.unused_candidates}
    more = registry.generate_candidates(200)
    assert all(tuple(c.phonemes) != tuple(candidates[0].phonemes) for c in more)


def test_registry_rejects_duplicate_reservations():
    registry = RootRegistry(sample_phonology(), seed=1)
    registry.reserve(list("sava"), owner_id="lex_hello", owner_kind="user_word")
    registry.reserve(list("sava"), owner_id="lex_hello", owner_kind="user_word")  # same owner: fine
    with pytest.raises(DuplicateFormError):
        registry.reserve(list("sava"), owner_id="lex_other", owner_kind="root")


def test_registry_accept_moves_candidate_into_language():
    registry = RootRegistry(sample_phonology(), seed=1)
    [candidate] = registry.generate_candidates(1)
    registry.accept(candidate.id, owner_id="lex_sea")
    assert registry.unused_candidates == []
    assert registry.is_reserved(candidate.phonemes)


def test_tiny_inventory_fails_loudly():
    tiny = Phonology(phonemes=[Phoneme.from_grapheme(g) for g in "pa"], syllable_patterns=["CV"])
    with pytest.raises(GenerationError, match="unique roots"):
        RootRegistry(tiny, seed=1).generate_candidates(10)
