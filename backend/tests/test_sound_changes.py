import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from app.evolution import templates
from app.evolution.sound_change_engine import (
    SoundChangeError, apply_changes, apply_sound_change, apply_to_word, resolve_sound,
)
from app.generation.morphological_engine import MorphologyEngine
from app.generation.phoneme_generator import parse_spelling
from app.generation.root_generator import RootRegistry
from app.models.civilization import Civilization
from app.models.grammar import Grammar, MorphologicalType, WordOrder
from app.models.language import Language, LanguageState, LanguageVersion
from app.models.lexicon import PartOfSpeech
from app.models.morphology import DerivationalRule, Morpheme, MorphemeType
from app.models.phonology import Phoneme, Phonology, Position, PositionRestriction
from app.models.sound_change import ContextItem, SoundChange, SoundMatcher
from app.models.sounds import SOUNDS
from app.services.dictionary_view import DictionaryView
from app.validation.phonology_validator import validate_phoneme_sequence
from app.validation.morphology_validator import validate_lexicon

PHONOLOGY = Phonology(
    phonemes=[Phoneme.from_grapheme(g) for g in [*"ptkbdgmnslrh", "sh", *"aeiou"]],
    syllable_patterns=["V", "CV", "CVC"],
    restrictions=[PositionRestriction(phoneme_id="h", forbidden_positions=[Position.CODA])],
)
ROOTS = {"person": "kina", "house": "kasa", "fire": "tuki", "stone": "gemo", "water": "apa",
         "sun": "sol", "hand": "mano", "dog": "ka", "night": "hasa", "fish": "lutu", "path": "sika"}


def build_language() -> LanguageState:
    civ = Civilization(name="Test", founding_year=-1000, geography="coastal", economy="mercantile")
    language = Language(civilization_id=civ.id, name="Testish", founding_year=-1000)
    version = LanguageVersion(language_id=language.id, valid_from_year=-1000, phonology=PHONOLOGY,
                              grammar=Grammar(word_order=WordOrder.SOV,
                                              morphological_type=MorphologicalType.AGGLUTINATIVE))
    suffix = Morpheme(language_id=language.id, phonemes=["k", "i"], type=MorphemeType.SUFFIX, gloss="small")
    rule = DerivationalRule(id="diminutive", name="Diminutive", meaning="small X", morpheme_id=suffix.id,
                            input_parts_of_speech=[PartOfSpeech.NOUN], output_part_of_speech=PartOfSpeech.NOUN)
    engine = MorphologyEngine(language_id=language.id, version_id=version.id, phonology=PHONOLOGY,
                              registry=RootRegistry(PHONOLOGY, seed=1), affixes=[suffix], rules=[rule])
    ids = {c: engine.add_root(c, c, PartOfSpeech.NOUN, parse_spelling(w, PHONOLOGY)).id for c, w in ROOTS.items()}
    engine.derive(ids["house"], "diminutive", "hut", "hut")      # kasa + ki
    engine.derive(ids["sun"], "diminutive", "star", "star")      # sol + ki
    return LanguageState(civilization=civ, language=language, version=version,
                         morphemes=list(engine.morphemes.values()), derivational_rules=[rule],
                         lexemes=list(engine.lexemes.values()), forms=list(engine.forms.values()))


def words(state: LanguageState) -> dict[str, str]:
    return {lx.concept_id: state.form_of(lx.id).orthography for lx in state.lexemes}


def ipa(text: str) -> list[str]:
    return parse_spelling(text, PHONOLOGY)


# --- rules ---------------------------------------------------------------------

def test_feature_outputs_pick_the_closest_sound():
    change = templates.palatalization()
    assert resolve_sound(change, "k") == "tʃ"
    assert resolve_sound(change, "ɡ") == "dʒ"


def test_sounds_without_a_counterpart_are_left_alone():
    # There is no voiced uvular stop, so voicing leaves /q/ as it is, and the preview says so.
    voicing = templates.intervocalic_voicing()
    assert resolve_sound(voicing, "q") == "q"
    phonology = PHONOLOGY.model_copy(update={"phonemes": [*PHONOLOGY.phonemes, Phoneme.from_grapheme("q")]})
    state = build_language()
    state = state.model_copy(update={"version": state.version.model_copy(update={"phonology": phonology})})
    assert apply_sound_change(state, voicing, year=-900).preview.steps[0].unchanged_sounds == ["q"]


def test_no_two_sounds_share_all_features():
    """Feature outputs are unambiguous only if every sound has a distinct feature set."""
    keys = [tuple(sorted((k, str(v)) for k, v in s.feature_values().items() if k != "sibilant"))
            for s in SOUNDS.values()]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("kwargs, message", [
    ({"target": SoundMatcher(ipa=["k"])}, "say what happens"),
    ({"target": SoundMatcher(ipa=["k"]), "delete": True, "output_mapping": {"k": "g"}}, "a rule does one thing"),
    ({"output_features": {"voiced": True}}, "target is required"),
    ({"target": SoundMatcher(ipa=["k"]), "output_features": {"class": "V"}}, "class"),
    ({"target": SoundMatcher(ipa=["k"]), "output_features": {"colour": "red"}}, "unknown feature"),
    ({"target": SoundMatcher(ipa=["k"]), "output_features": {"voiced": "yes"}}, "voiced"),
    ({"output_mapping": {"k": "a"}}, "consonant <-> vowel"),
    ({"output_mapping": {"k": "kk"}}, "unknown sound"),
])
def test_invalid_rules_are_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        SoundChange(name="bad", **kwargs)


def test_rules_round_trip_through_json():
    for make in templates.TEMPLATES.values():
        change = make()
        assert SoundChange.model_validate(json.loads(change.model_dump_json())) == change


# --- applying to one word ------------------------------------------------------

def test_application_is_simultaneous():
    # Both k's are checked against the original word.
    assert apply_changes([templates.palatalization()], ipa("kiki")) == ["tʃ", "i", "tʃ", "i"]


def test_environment_must_match():
    change = templates.palatalization()
    assert apply_changes([change], ipa("kasa")) == ipa("kasa")       # not before a front vowel
    assert apply_changes([change], ipa("kina"))[0] == "tʃ"


def test_consonant_runs_let_rules_see_the_next_vowel():
    # i-umlaut looks past consonants: "mano" is unaffected, "hasi" fronts its a.
    umlaut = templates.i_umlaut()
    assert apply_changes([umlaut], ipa("mano")) == ipa("mano")
    assert apply_changes([umlaut], ipa("hasi")) == ["h", "e", "s", "i"]
    assert apply_changes([umlaut], ipa("sulmi")) == ["s", "y", "l", "m", "i"]


def test_final_vowel_loss_spares_one_syllable_words():
    loss = templates.final_vowel_loss()
    assert apply_changes([loss], ipa("kasa")) == ipa("kas")
    assert apply_changes([loss], ipa("ka")) == ipa("ka")


def test_deletion_marks_positions():
    assert apply_to_word(templates.h_loss(), ipa("hasa")) == [[], ["a"], ["s"], ["a"]]


def test_changes_apply_in_historical_order():
    # Voicing first feeds spirantization: t -> d -> ð. The other order gives only d.
    voicing, spirant = templates.intervocalic_voicing(), templates.spirantization()
    assert apply_changes([voicing, spirant], ipa("lutu")) == ["l", "u", "ð", "u"]
    assert apply_changes([spirant, voicing], ipa("lutu")) == ["l", "u", "d", "u"]


# --- applying to a language ----------------------------------------------------

def test_preview_reports_impact_and_builds_next_version():
    state = build_language()
    result = apply_sound_change(state, templates.palatalization(), year=-700)
    p = result.preview
    assert p.ok and result.state is not None
    assert {c.concept_id: c.after for c in p.changed} == {
        "person": "china", "fire": "tuchi", "stone": "jemo", "hut": "kasachi", "star": "solchi"}
    assert p.steps[0].sound_mapping == {"k": "tʃ", "ɡ": "dʒ"}
    assert p.new_sounds == ["dʒ", "tʃ"]
    assert p.lost_sounds == ["ɡ"]   # g only ever stood before e; k survives in kasa and ka

    new = result.state
    assert new.version.valid_from_year == -700 and new.version.predecessor_version_id == state.version.id
    assert new.previous_versions == [state.version]
    assert new.version.phonology.phoneme("tʃ").grapheme == "ch"
    # The old forms are kept as history; morphemes keep their founding shape.
    assert len(new.forms) == 2 * len(state.forms)
    assert new.morphemes == state.morphemes
    assert words(state)["person"] == "kina" and words(new)["person"] == "china"


def test_regular_change_reaches_across_morpheme_boundaries():
    """The suffix -ki palatalizes in every word, wherever it attaches."""
    new = apply_sound_change(build_language(), templates.palatalization(), year=-700).state
    hut = next(lx for lx in new.lexemes if lx.concept_id == "hut")
    segments = [new.version.phonology.spell(s.phonemes) for s in new.form_of(hut.id).segments]
    assert segments == ["kasa", "chi"]


def test_evolved_language_still_validates():
    state = build_language()
    for change in [templates.palatalization(), templates.intervocalic_voicing(), templates.h_loss()]:
        state = apply_sound_change(state, change, year=state.version.valid_from_year + 100).state
    result = validate_lexicon(state)
    assert result.ok, [i.message for i in result.errors]
    assert len(state.previous_versions) == 3
    # The dictionary view works on the evolved language.
    view = DictionaryView(state)
    assert all(view.detail(lx.id) for lx in state.lexemes)


def test_validator_catches_a_form_that_skipped_a_change():
    state = apply_sound_change(build_language(), templates.palatalization(), year=-700).state
    person = next(lx for lx in state.lexemes if lx.concept_id == "person")
    forms = [f.model_copy(update={"phonemes": ipa("kina"), "orthography": "kina"})
             if f.lexeme_id == person.id and f.language_version_id == state.version.id else f
             for f in state.forms]
    broken = state.model_copy(update={"forms": forms})
    assert "FORM_MISMATCH" in [i.code for i in validate_lexicon(broken).errors]


def test_merger_loses_a_sound_and_creates_homophones():
    state = build_language()  # kasa (house) and hasa (night)
    result = apply_sound_change(state, templates.h_loss(), year=-500)
    assert result.preview.lost_sounds == ["h"]
    assert not result.state.version.phonology.has_phoneme("h")
    # h-loss turns hasa into asa: no clash. Merging k into h does clash.
    clash = apply_sound_change(state, templates.merger("k", "h"), year=-500)
    assert ["house", "night"] in [h.concept_ids for h in clash.preview.new_homophones]
    # Homophones from a merger are a warning, not an error.
    issues = validate_lexicon(clash.state).issues
    assert any(i.code == "DUPLICATE_FORM" and i.severity == "warning" for i in issues)
    assert validate_lexicon(clash.state).ok


def test_sound_rules_adapt_to_the_new_words():
    state = build_language()   # h may not end a syllable
    # s -> h never puts h in a coda here (sol -> hol, kasa -> kaha), so the restriction survives.
    onset_only = apply_sound_change(state, templates.merger("s", "h"), year=-500)
    assert onset_only.preview.ok and onset_only.state.version.phonology.restrictions
    # l -> h does (sol -> soh), so the restriction no longer describes the language and is dropped.
    coda = apply_sound_change(state, templates.merger("l", "h"), year=-500)
    assert coda.preview.ok
    assert any("may now appear in coda" in n for n in coda.preview.phonotactic_changes)
    assert not coda.state.version.phonology.restrictions


def test_clusters_that_break_sonority_block_the_change():
    """Syncope (losing vowels between consonants) makes clusters; only sonority-legal ones survive."""
    syncope = SoundChange(name="syncope", target=SoundMatcher(features={"class": "V"}), delete=True,
                          left=[ContextItem(kind="segment", matcher=SoundMatcher(features={"class": "C"}))],
                          right=[ContextItem(kind="segment", matcher=SoundMatcher(features={"class": "C"}))])
    result = apply_sound_change(build_language(), syncope, year=-500)
    assert not result.preview.ok and result.state is None
    illegal = {w.concept_id: w.form for w in result.preview.illegal}
    # lutu -> ltu and mano -> mno break sonority; kina -> kna (rising, like Old English cniht) does not.
    assert illegal["fish"] == "ltu" and illegal["hand"] == "mno"
    assert "person" not in illegal
    assert any("ILLEGAL_CLUSTER" in p or "cannot begin" in p for p in result.preview.illegal[0].problems)


def test_final_vowel_loss_creates_codas_and_new_shapes():
    loss = templates.final_vowel_loss()
    state = build_language()
    result = apply_sound_change(state, loss, year=-500)
    assert result.preview.ok, result.preview.illegal
    words_after = words(result.state)
    assert words_after["house"] == "kas" and words_after["hut"] == "kasak" and words_after["dog"] == "ka"


def test_final_vowel_loss_can_spare_clusters():
    loss = templates.final_vowel_loss(after_single_consonant=True)
    assert apply_changes([loss], ipa("kasa")) == ipa("kas")
    assert apply_changes([loss], ipa("kanta")) == ipa("kanta")   # n t _ : kept
    assert apply_changes([loss], ipa("ka")) == ipa("ka")


def test_new_cluster_shapes_are_added_only_when_needed():
    # kasaki -> kaski: "s.ki" parses with the existing CVC + CV, so no CCV is added.
    syncope = SoundChange(name="medial syncope", target=SoundMatcher(ipa=["a"]), delete=True,
                          left=[ContextItem(kind="segment", matcher=SoundMatcher(ipa=["s"]))],
                          right=[ContextItem(kind="segment", matcher=SoundMatcher(ipa=["k"]))])
    result = apply_sound_change(build_language(), syncope, year=-500)
    assert result.preview.ok and words(result.state)["hut"] == "kaski"
    assert result.state.version.phonology.syllable_patterns == PHONOLOGY.syllable_patterns
    # sol -> sl... no: a word-initial cluster does need a new shape.
    initial = SoundChange(name="initial syncope", target=SoundMatcher(ipa=["i"]), delete=True,
                          left=[ContextItem(kind="boundary"), ContextItem(kind="segment", matcher=SoundMatcher(ipa=["k"]))])
    result = apply_sound_change(build_language(), initial, year=-500)   # kina -> kna
    assert words(result.state)["person"] == "kna"
    assert "CCV" in result.state.version.phonology.syllable_patterns
    assert any("new syllable shapes" in n for n in result.preview.phonotactic_changes)


def test_s_clusters_switch_on_when_a_change_creates_them():
    # sika -> ska: s before a stop breaks sonority, but in the way s-clusters may.
    change = SoundChange(name="initial i loss", target=SoundMatcher(ipa=["i"]), delete=True,
                         left=[ContextItem(kind="boundary"), ContextItem(kind="segment", matcher=SoundMatcher(ipa=["s"]))])
    result = apply_sound_change(build_language(), change, year=-500)
    assert result.preview.ok and words(result.state)["path"] == "ska"
    ph = result.state.version.phonology
    assert ph.allow_s_clusters and "CCV" in ph.syllable_patterns
    assert any("s may now cluster" in n for n in result.preview.phonotactic_changes)


def test_sonority_rules():
    strict = PHONOLOGY.model_copy(update={"syllable_patterns": ["CV", "CCV", "CVC", "CCVC", "CVCC"]})
    loose = strict.model_copy(update={"allow_s_clusters": True})
    for word in ["plano", "kalp", "trin"]:
        assert validate_phoneme_sequence(ipa(word), strict).ok, word
    for word in ["lpano", "kapl", "ttan"]:   # falling onset, rising coda, plateau
        assert not validate_phoneme_sequence(ipa(word), strict).ok, word
    assert not validate_phoneme_sequence(ipa("staka"), strict).ok
    assert validate_phoneme_sequence(ipa("staka"), loose).ok
    assert validate_phoneme_sequence(ipa("taks"), loose).ok


def test_new_sound_spelling_can_be_chosen():
    result = apply_sound_change(build_language(), templates.palatalization(), year=-700, spellings={"tʃ": "č"})
    assert {c.concept_id: c.after for c in result.preview.changed}["person"] == "čina"


def test_year_must_move_forward():
    with pytest.raises(SoundChangeError, match="not after"):
        apply_sound_change(build_language(), templates.h_loss(), year=-1000)


@settings(max_examples=60, deadline=None)
@given(order=st.permutations(list(templates.TEMPLATES)), count=st.integers(min_value=1, max_value=4))
def test_any_sequence_of_legal_changes_keeps_the_lexicon_valid(order, count):
    """Whatever the engine accepts, the validator must agree with (the two can't drift apart)."""
    state = build_language()
    for name in order[:count]:
        result = apply_sound_change(state, templates.TEMPLATES[name](), year=state.version.valid_from_year + 50)
        if result.state is not None:
            state = result.state
    result = validate_lexicon(state)
    assert result.ok, [i.message for i in result.errors]
