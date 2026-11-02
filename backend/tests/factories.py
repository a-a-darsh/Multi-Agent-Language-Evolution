"""Hand-built fixtures. `sample_language` is the Phase 1 reference language."""

from app.models.civilization import Civilization
from app.models.grammar import Grammar, MorphologicalType, WordOrder
from app.models.language import Language, LanguageState, LanguageVersion
from app.models.lexicon import Derivation, LexemeOrigin, Lexeme, LexemeForm, LexicalSense, PartOfSpeech
from app.models.morphology import DerivationalRule, Morpheme, MorphemeOrigin, MorphemeType
from app.models.phonology import Phoneme, Phonology, Position, PositionRestriction


def sample_phonology(**overrides) -> Phonology:
    fields = dict(
        phonemes=[Phoneme.from_grapheme(g) for g in "ptkmnslrvaeiou"],
        syllable_patterns=["CV", "CVC"],
        restrictions=[PositionRestriction(phoneme_id="v", forbidden_positions=[Position.CODA])],
    )
    fields.update(overrides)
    return Phonology(**fields)


def sample_language() -> LanguageState:
    civ = Civilization(name="Savari", founding_year=-1200, geography="coastal",
                       economy="mercantile", cultural_traits=["seafaring"])
    language = Language(civilization_id=civ.id, name="Old Savari", founding_year=-1200)
    version = LanguageVersion(
        language_id=language.id,
        valid_from_year=-1200,
        phonology=sample_phonology(),
        grammar=Grammar(word_order=WordOrder.SOV, morphological_type=MorphologicalType.AGGLUTINATIVE),
    )

    nal = Morpheme(language_id=language.id, phonemes=list("nal"), type=MorphemeType.ROOT, gloss="sea")
    ka = Morpheme(language_id=language.id, phonemes=list("ka"), type=MorphemeType.SUFFIX,
                  gloss="person associated with")
    sava = Morpheme(language_id=language.id, phonemes=list("sava"), type=MorphemeType.ROOT,
                    gloss="hello", origin=MorphemeOrigin.USER_DEFINED)
    rule = DerivationalRule(id="associated_person", name="Associated person",
                            meaning="person associated with a concept", morpheme_id=ka.id,
                            input_parts_of_speech=[PartOfSpeech.NOUN],
                            output_part_of_speech=PartOfSpeech.NOUN)

    sea = Lexeme(language_id=language.id, concept_id="sea", part_of_speech=PartOfSpeech.NOUN,
                 origin=LexemeOrigin.GENERATED_ROOT, morpheme_ids=[nal.id],
                 senses=[LexicalSense(concept_id="sea", gloss="sea")])
    hello = Lexeme(language_id=language.id, concept_id="hello", part_of_speech=PartOfSpeech.INTERJECTION,
                   origin=LexemeOrigin.USER_DEFINED, morpheme_ids=[sava.id],
                   senses=[LexicalSense(concept_id="hello", gloss="hello")])
    sailor_id = "lex_sailor"
    sailor = Lexeme(id=sailor_id, language_id=language.id, concept_id="sailor",
                    part_of_speech=PartOfSpeech.NOUN, origin=LexemeOrigin.DERIVED,
                    morpheme_ids=[nal.id, ka.id],
                    senses=[LexicalSense(concept_id="sailor", gloss="sailor")],
                    derivation=Derivation(output_lexeme_id=sailor_id, base_lexeme_id=sea.id,
                                          rule_id=rule.id, applied_morpheme_ids=[ka.id]))

    forms = [
        LexemeForm(lexeme_id=lx.id, language_version_id=version.id, phonemes=list(text), orthography=text)
        for lx, text in [(sea, "nal"), (hello, "sava"), (sailor, "nalka")]
    ]
    return LanguageState(civilization=civ, language=language, version=version,
                         morphemes=[nal, ka, sava], derivational_rules=[rule],
                         lexemes=[sea, hello, sailor], forms=forms)
