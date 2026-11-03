"""Command-line entry points for trying agents against a real model.

    python -m uv run python -m app.cli design --example coastal_trading_sov
    python -m uv run python -m app.cli design --example evidentiality --model <ollama model>
    python -m uv run python -m app.cli graph
    python -m uv run python -m app.cli evolve-agent --language Varish --years 200 --context "..."   # LLM agent
    python -m uv run python -m app.cli evolve --language Varish palatalization final_vowel_loss+cluster_epenthesis   # add --save to keep

Uses the LLM saved in the UI's setup screen unless --model is given (Ollama).
Each run is saved to backend/.local/runs/ with every prompt and reply.
"""

import argparse
import json
import sys
from pathlib import Path

from app.agents.language_designer import DesignOutcome, build_design_graph, run_language_design
from app.data import load_example_requests
from app.llm.chat_model import build_chat_model
from app.llm.settings import LLMProvider, LLMSettings, SettingsStore
from app.llm.structured import LangChainStructuredLLM, ScriptedLLM

RUNS_DIR = Path(__file__).resolve().parents[1] / ".local" / "runs"


def _settings(model: str | None) -> LLMSettings:
    if model:
        return LLMSettings(provider=LLMProvider.LOCAL, model=model)
    saved = SettingsStore.default().load()
    if saved is None:
        sys.exit("No LLM configured. Use the setup screen in the UI, or pass --model <ollama model>.")
    return saved


def _print_outcome(outcome: DesignOutcome) -> None:
    print("LLM calls: " + ", ".join(f"{c.step} {c.seconds}s" for c in outcome.llm_calls))
    for i, attempt in enumerate(outcome.attempts, 1):
        status = "ok" if not attempt.issues else f"{len(attempt.issues)} finding(s)"
        print(f"  {i}. {attempt.step}: {status}")
        for issue in attempt.issues:
            print(f"      [{issue.code}] {issue.message}")
    print(f"\nStatus: {outcome.status.upper()}")
    for w in outcome.warnings:
        print(f"  warning [{w.code}] {w.message}")
    design = outcome.design
    if design is None:
        return
    ph, g = design.phonology, design.grammar
    print(f"\n{design.name}\n{'=' * len(design.name)}\n{design.summary}\n")
    print(f"Consonants: {' '.join(p.grapheme for p in ph.consonants)}")
    print(f"Vowels:     {' '.join(p.grapheme for p in ph.vowels)}")
    print(f"Syllables:  {', '.join(ph.syllable_patterns)}   hiatus allowed: {ph.allow_hiatus}")
    for r in ph.restrictions:
        print(f"Restriction: {ph.spell([r.phoneme_id])} not in {[p.value for p in r.forbidden_positions]}")
    print(f"Grammar:    {g.word_order.value}, {g.morphological_type.value}, {g.adposition_type.value if g.adposition_type else '-'}")
    for f in g.features:
        print(f"Feature:    {f.name} - {f.realization}")
    affixes = {m.id: m for m in design.affixes}
    print("Affixes:")
    for rule in design.derivational_rules:
        m = affixes[rule.morpheme_id]
        form = ph.spell(m.phonemes)
        shown = f"{form}-" if m.type.value == "prefix" else f"-{form}"
        print(f"  {shown:<8} {rule.name}: {rule.meaning}")
    if design.boundary_repair:
        link = design.boundary_repair
        print(f"Linking:    vowel={ph.spell([link.linking_vowel]) if link.linking_vowel else None} "
              f"consonant={ph.spell([link.linking_consonant]) if link.linking_consonant else None}")


def design_command(args) -> None:
    examples = load_example_requests()
    if args.example not in examples:
        sys.exit(f"Unknown example. Choose from: {', '.join(examples)}")
    settings = _settings(args.model)
    llm = LangChainStructuredLLM(build_chat_model(settings, temperature=args.temperature), settings.model)
    # Same model, different role: the critic runs at temperature 0 for consistent judgments.
    critic = None if args.no_critic else LangChainStructuredLLM(
        build_chat_model(settings, temperature=0, max_tokens=1500), settings.model)
    print(f"Designing '{args.example}' with {settings.provider.value}:{settings.model} ...\n")
    outcome = run_language_design(examples[args.example], llm, critic_llm=critic)
    _print_outcome(outcome)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"design-{args.example}-{outcome.llm_calls[0].created_at:%Y%m%d-%H%M%S}.json" \
        if outcome.llm_calls else RUNS_DIR / f"design-{args.example}.json"
    path.write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nFull record (prompts, replies, issues): {path}")


def _print_dictionary(outcome) -> None:
    d = outcome.dictionary
    print("\nDictionary LLM calls: " + ", ".join(f"{c.step} {c.seconds}s" for c in d.llm_calls))
    for r in d.rounds:
        if r.issues:
            print(f"  {r.step}: {len(r.issues)} finding(s)")
            for i in r.issues[:8]:
                print(f"      [{i.code}] {i.message}")
    print(f"Status: {d.status.upper()}")
    for w in d.warnings:
        print(f"  warning [{w.code}] {w.message}")
    if d.state is None:
        return
    state = d.state
    form = {lx.id: state.form_of(lx.id).orthography for lx in state.lexemes}
    by_origin: dict[str, int] = {}
    for lx in state.lexemes:
        by_origin[lx.origin.value] = by_origin.get(lx.origin.value, 0) + 1
    print(f"\n{len(state.lexemes)} entries: " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in by_origin.items()))

    children: dict[str, list] = {}
    for lx in state.lexemes:
        for base in lx.base_lexeme_ids:  # a compound appears under both of its parts
            children.setdefault(base, []).append(lx)
    roots = sorted((lx for lx in state.lexemes if lx.id in children and not lx.base_lexeme_ids),
                   key=lambda lx: -len(children[lx.id]))
    concept = {lx.id: lx.concept_id for lx in state.lexemes}

    def show(lx, indent: int) -> None:
        how = (f"  (+{lx.derivation.rule_id})" if lx.derivation else
               f"  (= {' + '.join(concept[p] for p in lx.compound.part_lexeme_ids)})" if lx.compound else "")
        print(f"{'    ' * indent}{form[lx.id]:<14} {lx.senses[0].gloss}{how}")
        for child in children.get(lx.id, []):
            show(child, indent + 1)

    print("\nWord families:")
    for root in roots[:12]:
        show(root, 1)


def generate_command(args) -> None:
    from app.agents.pipeline import agent_llms_from_settings, run_generation

    examples = load_example_requests()
    if args.example not in examples:
        sys.exit(f"Unknown example. Choose from: {', '.join(examples)}")
    settings = _settings(args.model)
    llms = agent_llms_from_settings(settings, use_critic=not args.no_critic)
    print(f"Generating '{args.example}' with {settings.provider.value}:{settings.model} ...\n")
    outcome = run_generation(examples[args.example], llms, on_step=lambda step: print(f"  ... {step}"))
    _print_outcome(outcome.design)
    if outcome.dictionary:
        _print_dictionary(outcome)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"generate-{args.example}-{outcome.design.llm_calls[0].created_at:%Y%m%d-%H%M%S}.json"
    path.write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nFull record: {path}")


def _find_language(name: str):
    from app.persistence.database import make_session_factory
    from app.persistence.repository import Repository

    repo = Repository(make_session_factory())
    match = next((lang for lang in repo.list_languages()
                  if name in (lang.id, lang.name) or lang.name.lower() == name.lower()), None)
    if match is None:
        sys.exit(f"No saved language {name!r}. Saved: {', '.join(lang.name for lang in repo.list_languages())}")
    return repo, match


def evolve_agent_command(args) -> None:
    """Let the Linguistic Evolution Agent evolve a saved language. Saved only with --save."""
    from app.agents.evolution_agent import run_evolution_agent
    from app.llm.tool_calling import LangChainToolLLM

    repo, match = _find_language(args.language)
    state = repo.get_language(match.id).state
    settings = _settings(args.model)
    llm = LangChainToolLLM(build_chat_model(settings, temperature=args.temperature, max_tokens=2048), settings.model)
    print(f"Evolving {match.name} (year {state.version.valid_from_year}) over {args.years} years with "
          f"{settings.provider.value}:{settings.model} ...")
    if args.context:
        print(f"Context: {args.context}")

    outcome = run_evolution_agent(state, llm, years=args.years, context=args.context, max_llm_calls=args.max_calls,
                                  on_step=lambda node: print(f"  ... {'thinking' if node == 'act' else node}",
                                                             flush=True))
    for use in outcome.tool_log:
        args_text = json.dumps(use.args, ensure_ascii=False)
        print(f"\n> {use.tool}({args_text if len(args_text) < 400 else args_text[:400] + '...'})")
        print("  " + use.result.replace("\n", "\n  "))
    print(f"\nStatus: {outcome.status.upper()} after {len(outcome.llm_calls)} model calls, "
          f"{outcome.previews} previews, {sum(c.seconds for c in outcome.llm_calls):.0f}s")
    for issue in outcome.issues:
        print(f"  issue: {issue}")
    if outcome.rationale:
        print(f"Rationale: {outcome.rationale}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = outcome.llm_calls[0].created_at if outcome.llm_calls else None
    path = RUNS_DIR / f"evolve-{match.name}-{stamp:%Y%m%d-%H%M%S}.json"
    path.write_text(outcome.model_dump_json(indent=2, exclude={"state"}), encoding="utf-8")
    print(f"Full record: {path}")
    if args.save and outcome.state is not None:
        repo.save_language_state(outcome.state, expected_version_id=state.version.id)
        print(f"Saved: {match.name} is now at year {outcome.state.version.valid_from_year}.")


def evolve_command(args) -> None:
    """Preview sound changes on a saved language, in order. Saved only with --save."""
    from app.evolution.sound_change_engine import apply_sound_change
    from app.evolution.templates import TEMPLATES
    from app.persistence.database import make_session_factory
    from app.persistence.repository import Repository

    if unknown := [n for step in args.changes for n in step.split("+") if n not in TEMPLATES]:
        sys.exit(f"Unknown change(s) {unknown}. Choose from: {', '.join(TEMPLATES)}")
    repo, match = _find_language(args.language)
    state = repo.get_language(match.id).state
    saved_from = state.version.id
    for name in args.changes:
        result = apply_sound_change(state, [TEMPLATES[n]() for n in name.split("+")],
                                    year=state.version.valid_from_year + args.years)
        p = result.preview
        old_ph, new_ph = state.version.phonology, (result.state or state).version.phonology
        print(f"\n== {' + '.join(s.change.name for s in p.steps)}")
        for step in p.steps:
            mapping = ", ".join([f"{old_ph.spell([a])} -> {new_ph.spell([b]) if b else '(lost)'}"
                                 for a, b in step.sound_mapping.items()]
                                + [f"+{new_ph.spell([s])}" for s in step.inserted])
            print(f"   {step.change.name}: {step.change.description}; {step.words_changed} words; "
                  f"sounds: {mapping or 'none'}")
        print(f"   {len(p.changed)}/{p.total_words} words change in all")
        if p.new_sounds or p.lost_sounds:
            print(f"   new sounds: {p.new_sounds or '-'}   lost sounds: {p.lost_sounds or '-'}")
        for note in p.phonotactic_changes:
            print(f"   rule change: {note}")
        for c in p.changed[:args.show]:
            print(f"     {c.concept_id:<16} {c.before:<14} -> {c.after}")
        if len(p.changed) > args.show:
            print(f"     ... and {len(p.changed) - args.show} more")
        for h in p.new_homophones:
            print(f"   homophones: {h.form} = {', '.join(h.concept_ids)}")
        if not p.ok:
            print(f"   REFUSED: {len(p.illegal)} word(s) would break the sound rules, e.g.")
            for w in p.illegal[:3]:
                print(f"     {w.concept_id}: {w.form!r} - {'; '.join(w.problems)}")
            print("   (later changes are previewed on the language without this one)")
            continue
        state = result.state
    if args.save and state.version.id != saved_from:
        repo.save_language_state(state, expected_version_id=saved_from)
        print(f"\nSaved: {match.name} is now at year {state.version.valid_from_year} "
              f"({len(state.previous_versions) + 1} versions).")


def graph_command(_) -> None:
    # Mermaid source; paste into https://mermaid.live to see the diagram.
    print(build_design_graph(ScriptedLLM([])).get_graph().draw_mermaid())


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(required=True)
    design = sub.add_parser("design", help="Run the Language Design Agent on an example request")
    design.add_argument("--example", default="coastal_trading_sov")
    design.add_argument("--model", help="Ollama model to use instead of the saved settings")
    design.add_argument("--temperature", type=float, default=0.7)
    design.add_argument("--no-critic", action="store_true", help="Skip the LLM critic step")
    design.set_defaults(func=design_command)
    gen = sub.add_parser("generate", help="Design a language and build its dictionary")
    gen.add_argument("--example", default="coastal_trading_sov")
    gen.add_argument("--model", help="Ollama model to use instead of the saved settings")
    gen.add_argument("--no-critic", action="store_true")
    gen.set_defaults(func=generate_command)
    evolve = sub.add_parser("evolve", help="Preview sound changes on a saved language (--save to keep them)")
    evolve.add_argument("--language", required=True, help="Saved language name or id")
    evolve.add_argument("--years", type=int, default=100, help="Years between changes")
    evolve.add_argument("--show", type=int, default=12, help="Changed words to list per change")
    evolve.add_argument("--save", action="store_true", help="Save the accepted steps to the language")
    evolve.add_argument("changes", nargs="+",
                        help="Template names, applied in order; join with + for one step (a+b)")
    evolve.set_defaults(func=evolve_command)
    agent = sub.add_parser("evolve-agent", help="Let the evolution agent evolve a saved language (--save to keep)")
    agent.add_argument("--language", required=True, help="Saved language name or id")
    agent.add_argument("--years", type=int, default=200)
    agent.add_argument("--context", help="Historical context, e.g. 'two centuries of isolation in the mountains'")
    agent.add_argument("--model", help="Ollama model to use instead of the saved settings")
    agent.add_argument("--temperature", type=float, default=0.4)
    agent.add_argument("--max-calls", type=int, default=16)
    agent.add_argument("--save", action="store_true")
    agent.set_defaults(func=evolve_agent_command)
    sub.add_parser("graph", help="Print the design workflow as a Mermaid diagram").set_defaults(func=graph_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
