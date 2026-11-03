"""Insertion, multi-rule steps, exceptions, coining words in evolved languages, and saving them."""

import pytest

from app.evolution import templates
from app.evolution.history import apply_changes, realize
from app.evolution.lexicon import CoinageError, coin_root, derive_word
from app.evolution.sound_change_engine import SoundChangeError, apply_sound_change
from app.generation.morphological_engine import MorphologyEngine
from app.generation.phoneme_generator import parse_spelling
from app.generation.root_generator import RootRegistry
from app.models.civilization import Civilization, World
from app.models.grammar import Grammar, MorphologicalType, WordOrder
from app.models.language import Language, LanguageState, LanguageVersion
from app.models.lexicon import PartOfSpeech
from app.models.morphology import BoundaryRepair, DerivationalRule, Morpheme, MorphemeType
from app.models.phonology import Phoneme, Phonology
from app.models.sound_change import SoundChange, SoundChangeException
from app.persistence.database import GenerationRow, LanguageRow, make_session_factory
from app.persistence.repository import Repository, StaleLanguageError
from app.services.dictionary_view import DictionaryView
from app.validation.morphology_validator import validate_lexicon
from tests.test_languages_api import PLAN, env, generate  # noqa: F401  (env is a fixture)
from tests.test_language_designer import good_draft

PHONOLOGY = Phonology(phonemes=[Phoneme.from_grapheme(g) for g in [*"ptkbdgmnslr", *"aeiou"]],
                      syllable_patterns=["V", "CV", "CVC"])


def build(roots: dict[str, str], repair: BoundaryRepair | None = None) -> LanguageState:
    civ = Civilization(name="Test", founding_year=-1000, geography="coastal", economy="mercantile")
    language = Language(civilization_id=civ.id, name="Testish", founding_year=-1000)
    version = LanguageVersion(language_id=language.id, valid_from_year=-1000, phonology=PHONOLOGY,
                              grammar=Grammar(word_order=WordOrder.SOV,
                                              morphological_type=MorphologicalType.AGGLUTINATIVE),
                              boundary_repair=repair)
    suffix = Morpheme(language_id=language.id, phonemes=["k", "i"], type=MorphemeType.SUFFIX, gloss="small")
    rule = DerivationalRule(id="diminutive", name="Diminutive", meaning="small X", morpheme_id=suffix.id,
                            input_parts_of_speech=[PartOfSpeech.NOUN], output_part_of_speech=PartOfSpeech.NOUN)
    engine = MorphologyEngine(language_id=language.id, version_id=version.id, phonology=PHONOLOGY,
                              registry=RootRegistry(PHONOLOGY, seed=1), affixes=[suffix], rules=[rule],
                              boundary_repair=repair)
    for concept, word in roots.items():
        engine.add_root(concept, concept, PartOfSpeech.NOUN, parse_spelling(word, PHONOLOGY))
    return LanguageState(civilization=civ, language=language, version=version,
                         morphemes=list(engine.morphemes.values()), derivational_rules=[rule],
                         lexemes=list(engine.lexemes.values()), forms=list(engine.forms.values()))


def words(state: LanguageState) -> dict[str, str]:
    return {lx.concept_id: state.form_of(lx.id).orthography for lx in state.lexemes}


def lexeme(state: LanguageState, concept: str):
    return next(lx for lx in state.lexemes if lx.concept_id == concept)


def ipa(text: str) -> list[str]:
    return parse_spelling(text, PHONOLOGY)


def assert_valid(state: LanguageState) -> None:
    result = validate_lexicon(state)
    assert result.ok, [i.message for i in result.errors]


ROOTS = {"earth": "kanra", "house": "kasa", "person": "kina", "light": "dobla", "now": "ado"}


# --- insertion ---------------------------------------------------------------------

def test_epenthesis_breaks_a_final_cluster():
    assert apply_changes([templates.cluster_epenthesis()], ipa("kanr")) == ipa("kaner")
    assert apply_changes([templates.cluster_epenthesis()], ipa("kast")) == ipa("kast")   # s t: falls, left alone
    assert apply_changes([templates.cluster_epenthesis(sonorants_only=False)], ipa("kast")) == ipa("kaset")


def test_prothesis_adds_a_vowel_before_s_clusters():
    assert apply_changes([templates.prothesis()], ipa("skola")) == ipa("eskola")
    assert apply_changes([templates.prothesis()], ipa("sola")) == ipa("sola")


def test_insertion_rules_are_validated():
    with pytest.raises(ValueError, match="no target"):
        SoundChange(name="x", insert="e", target=templates.VOWEL, right=[templates.BOUNDARY])
    with pytest.raises(ValueError, match="needs a left or right"):
        SoundChange(name="x", insert="e")


def test_inserted_sound_joins_a_segment():
    state = build(ROOTS)
    new = apply_sound_change(state, [templates.final_vowel_loss(), templates.cluster_epenthesis()], -800).state
    form = new.form_of(lexeme(new, "earth").id)
    assert form.orthography == "kaner"
    assert [new.version.phonology.spell(s.phonemes) for s in form.segments] == ["kaner"]


# --- multi-rule steps ----------------------------------------------------------------

def test_a_step_is_judged_only_at_its_end():
    state = build(ROOTS)
    alone = apply_sound_change(state, templates.final_vowel_loss(), -800)
    assert not alone.preview.ok   # kanr and dobl break sonority
    step = apply_sound_change(state, [templates.final_vowel_loss(), templates.cluster_epenthesis()], -800)
    assert step.preview.ok
    assert words(step.state) | {} == {"earth": "kaner", "house": "kas", "person": "kin", "light": "dobel",
                                      "now": "ad"}
    assert [s.words_changed for s in step.preview.steps] == [5, 2]
    assert step.preview.steps[1].inserted == ["e"]
    assert step.state.version.sound_changes == step.preview.changes   # one version, both rules
    assert_valid(step.state)


# --- exceptions ----------------------------------------------------------------------

def test_an_exception_keeps_one_word_out_of_a_change():
    state = build(ROOTS)
    loss = templates.final_vowel_loss()
    keep = [SoundChangeException(change_id=loss.id, lexeme_id=lexeme(state, "now").id,
                                 reason="very frequent word; resisted the change")]
    result = apply_sound_change(state, [loss, templates.cluster_epenthesis()], -800, exceptions=keep)
    assert result.preview.ok
    assert words(result.state)["now"] == "ado" and words(result.state)["house"] == "kas"
    assert result.preview.steps[0].exceptions == ["now"]
    assert result.state.version.exceptions == keep
    assert_valid(result.state)   # the validator replays the exception too


def test_exceptions_must_matter_and_be_few():
    state = build(ROOTS)
    loss = templates.final_vowel_loss()
    unaffected = SoundChange(name="h loss", output_mapping={"p": "f"})   # no word has p
    with pytest.raises(SoundChangeError, match="does not change it"):
        apply_sound_change(state, [loss, unaffected], -800, exceptions=[
            SoundChangeException(change_id=unaffected.id, lexeme_id=lexeme(state, "now").id, reason="because")])
    with pytest.raises(SoundChangeError, match="not in this step"):
        apply_sound_change(state, loss, -800, exceptions=[
            SoundChangeException(change_id="other", lexeme_id=lexeme(state, "now").id, reason="because")])
    too_many = [SoundChangeException(change_id=loss.id, lexeme_id=lx.id, reason="because") for lx in state.lexemes]
    with pytest.raises(SoundChangeError, match="too many"):
        apply_sound_change(state, loss, -800, exceptions=too_many)


def test_validator_notices_a_removed_exception():
    state = build(ROOTS)
    loss = templates.final_vowel_loss()
    keep = [SoundChangeException(change_id=loss.id, lexeme_id=lexeme(state, "now").id, reason="frequent")]
    new = apply_sound_change(state, [loss, templates.cluster_epenthesis()], -800, exceptions=keep).state
    forged = new.model_copy(update={"version": new.version.model_copy(update={"exceptions": []})})
    assert "FORM_MISMATCH" in [i.code for i in validate_lexicon(forged).errors]


# --- linking sounds evolve -------------------------------------------------------------

def test_boundary_repair_sounds_follow_sound_changes():
    state = build(ROOTS, repair=BoundaryRepair(linking_vowel="i"))
    new = apply_sound_change(state, SoundChange(name="i to e", output_mapping={"i": "e"}), -800).state
    assert new.version.boundary_repair.linking_vowel == "e"


# --- coining words ---------------------------------------------------------------------

def palatalized() -> LanguageState:
    state = build(ROOTS)
    state = derive_word(state, lexeme(state, "house").id, "diminutive", "hut", "hut")   # kasa + ki, founding era
    return apply_sound_change(state, templates.palatalization(), -700).state


def test_new_derivations_use_todays_shapes():
    state = palatalized()
    assert words(state)["person"] == "china" and words(state)["hut"] == "kasachi"
    # -ki is now -chi; a new diminutive of person is built from "china" + "chi".
    state = derive_word(state, lexeme(state, "person").id, "diminutive", "child", "child")
    child = state.form_of(lexeme(state, "child").id)
    assert child.orthography == "chinachi"
    assert [state.version.phonology.spell(s.phonemes) for s in child.segments] == ["china", "chi"]
    assert lexeme(state, "child").coined_in_version_id == state.version.id
    assert_valid(state)


def test_coined_words_take_part_in_later_changes():
    state = palatalized()
    state = derive_word(state, lexeme(state, "person").id, "diminutive", "child", "child")
    state = coin_root(state, "boat", "boat", PartOfSpeech.NOUN, spelling="sotu")
    state = apply_sound_change(state, templates.intervocalic_voicing(), -500).state
    assert words(state)["boat"] == "sodu"
    assert words(state)["child"] == "chinachi"   # ch is an affricate, not a stop: unaffected
    assert_valid(state)
    forms, problems = realize(state)
    assert problems.ok and forms[lexeme(state, "boat").id].phonemes == ipa("sodu")


def test_new_roots_use_the_current_spelling():
    state = palatalized()
    state = coin_root(state, "cheese", "cheese", PartOfSpeech.NOUN, spelling="chiro")
    assert words(state)["cheese"] == "chiro"
    with pytest.raises(CoinageError, match="no longer has|unsupported"):
        coin_root(state, "zebra", "zebra", PartOfSpeech.NOUN, spelling="zibu")   # z was never in the language


def test_generated_roots_obey_the_current_sound_system():
    state = palatalized()
    state = coin_root(state, "comet", "comet", PartOfSpeech.NOUN, seed=3)
    form = state.form_of(lexeme(state, "comet").id)
    assert lexeme(state, "comet").origin == "generated_root"
    assert_valid(state)
    assert form.phonemes not in [state.form_of(lx.id).phonemes for lx in state.lexemes[:-1]]


def test_coinage_refuses_homophones_and_duplicates():
    state = palatalized()
    with pytest.raises(CoinageError, match="already the word for"):
        coin_root(state, "shell", "shell", PartOfSpeech.NOUN, spelling="kasa")
    with pytest.raises(CoinageError, match="already has a word"):
        coin_root(state, "house", "house", PartOfSpeech.NOUN, spelling="lomo")


def test_dictionary_view_shows_coined_words():
    state = palatalized()
    state = derive_word(state, lexeme(state, "person").id, "diminutive", "child", "child")
    detail = DictionaryView(state).detail(lexeme(state, "child").id)
    assert detail.word == "chinachi" and detail.derived_from.concept_id == "person"
    assert [s.kind for s in detail.segments] == ["root", "suffix"]


# --- saving --------------------------------------------------------------------------------

def test_saving_an_evolved_language_and_refusing_stale_writes(tmp_path):
    repo = Repository(make_session_factory(tmp_path / "db.sqlite3"))
    state = build(ROOTS)
    world = World(seed=1, focal_civilization_id=state.civilization.id, civilizations=[state.civilization])
    with repo.sessions.begin() as s:
        s.add(GenerationRow(id="g", status="succeeded", request_json="{}"))
        s.add(LanguageRow(id=state.language.id, name="Testish", civilization_name="Test", entry_count=5,
                          derived_count=0, generation_id="g", state_json=state.model_dump_json(),
                          world_json=world.model_dump_json()))
    evolved = apply_sound_change(state, templates.palatalization(), -700).state
    repo.save_language_state(evolved, expected_version_id=state.version.id)
    loaded = repo.get_language(state.language.id).state
    assert loaded == evolved and words(loaded)["person"] == "china"
    assert_valid(loaded)
    with pytest.raises(StaleLanguageError):
        repo.save_language_state(evolved, expected_version_id=state.version.id)   # it moved on


# --- API -----------------------------------------------------------------------------------

def test_evolution_api(env):  # noqa: F811
    client, scripts, _ = env
    lang_id = generate(client, scripts, [good_draft(), PLAN])["language_id"]
    base = f"/api/languages/{lang_id}"
    overview = client.get(base).json()
    assert overview["version_count"] == 1
    vowel = overview["vowels"][0]["ipa"]
    step = {"changes": [{"rule": {"name": "vowel shift", "output_mapping": {vowel: "ɔ"}}}], "years": 150}

    templates_list = client.get("/api/sound-change-templates").json()
    assert {"palatalization", "cluster_epenthesis"} <= {t["name"] for t in templates_list}

    preview = client.post(f"{base}/sound-changes/preview", json=step).json()
    assert preview["ok"] and preview["preview"]["changed"]
    assert client.get(base).json()["version_count"] == 1   # preview saves nothing

    stale = client.post(f"{base}/sound-changes", json={**step, "expected_version_id": "old"})
    assert stale.status_code == 409
    committed = client.post(f"{base}/sound-changes", json=step).json()
    overview = client.get(base).json()
    assert overview["version_count"] == 2 and overview["current_year"] == committed["year"]

    entries = client.get(f"{base}/dictionary", params={"page_size": 200}).json()["entries"]
    assert any("ò" in e["word"] for e in entries)

    noun = next(e for e in entries if e["part_of_speech"] == "noun" and e["origin"] != "derived")
    rule_id = overview["affixes"][0]["rule_id"]
    derived = client.post(f"{base}/words", json={"concept_id": "new_thing", "gloss": "new thing",
                                                 "base_lexeme_id": noun["id"], "rule_id": rule_id})
    assert derived.status_code == 201, derived.json()
    assert derived.json()["derived_from"]["id"] == noun["id"]
    root = client.post(f"{base}/words", json={"concept_id": "comet", "gloss": "comet", "part_of_speech": "noun"})
    assert root.status_code == 201, root.json()
    duplicate = client.post(f"{base}/words", json={"concept_id": "comet", "gloss": "comet", "part_of_speech": "noun"})
    assert duplicate.status_code == 422

    history = client.get(f"{base}/history").json()
    assert [len(v["changes"]) for v in history] == [0, 1]
    assert set(history[1]["words_coined"]) == {"new_thing", "comet"}

    refused = client.post(f"{base}/sound-changes", json={"changes": [{"rule": {
        "name": "lose all vowels", "target": {"features": {"class": "V"}}, "delete": True,
        "right": [{"kind": "boundary"}]}}]})
    assert refused.status_code == 422 and refused.json()["error"]["code"] in {"SOUND_CHANGE_REFUSED"}


# --- property: any history replays exactly ----------------------------------------------------

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

ACTIONS = st.lists(st.one_of(
    st.tuples(st.just("step"), st.lists(st.sampled_from(sorted(templates.TEMPLATES)), min_size=1, max_size=3),
              st.booleans()),
    st.tuples(st.just("derive"), st.integers(min_value=0, max_value=50), st.just(None)),
    st.tuples(st.just("coin"), st.integers(min_value=0, max_value=1000), st.just(None)),
), min_size=1, max_size=6)


@settings(max_examples=40, deadline=None)
@given(actions=ACTIONS)
def test_any_mix_of_changes_exceptions_and_new_words_replays_exactly(actions):
    """Whatever the engine and coinage accept, history replay must reproduce every stored form."""
    state = build(ROOTS)
    for n, (kind, arg, flag) in enumerate(actions):
        try:
            if kind == "step":
                changes = [templates.TEMPLATES[name]() for name in arg]
                exceptions = []
                if flag:  # exempt the first word the first rule would change
                    first = next((lx for lx in state.lexemes
                                  if apply_changes(changes[:1], state.form_of(lx.id).phonemes)
                                  != state.form_of(lx.id).phonemes), None)
                    if first:
                        exceptions = [SoundChangeException(change_id=changes[0].id, lexeme_id=first.id,
                                                           reason="frequent word")]
                result = apply_sound_change(state, changes, state.version.valid_from_year + 50, exceptions=exceptions)
                state = result.state or state
            elif kind == "derive":
                bases = [lx for lx in state.lexemes if lx.derivation is None]
                state = derive_word(state, bases[arg % len(bases)].id, "diminutive", f"dim_{n}", "small")
            else:
                state = coin_root(state, f"new_{n}", "new", PartOfSpeech.NOUN, seed=arg)
        except (CoinageError, SoundChangeError):
            pass  # refusing is fine; accepting something invalid is not
    assert_valid(state)
