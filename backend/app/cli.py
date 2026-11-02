"""Command-line entry points for trying agents against a real model.

    python -m uv run python -m app.cli design --example coastal_trading_sov
    python -m uv run python -m app.cli design --example evidentiality --model <ollama model>
    python -m uv run python -m app.cli graph

Uses the LLM saved in the UI's setup screen unless --model is given (Ollama).
Each run is saved to backend/.local/runs/ with every prompt and reply.
"""

import argparse
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
        print(f"Restriction: {r.phoneme_id} not in {[p.value for p in r.forbidden_positions]}")
    print(f"Grammar:    {g.word_order.value}, {g.morphological_type.value}, {g.adposition_type.value if g.adposition_type else '-'}")
    for f in g.features:
        print(f"Feature:    {f.name} - {f.realization}")
    affixes = {m.id: m for m in design.affixes}
    print("Affixes:")
    for rule in design.derivational_rules:
        m = affixes[rule.morpheme_id]
        form = "".join(m.phonemes)
        shown = f"{form}-" if m.type.value == "prefix" else f"-{form}"
        print(f"  {shown:<8} {rule.name}: {rule.meaning}")
    if design.boundary_repair:
        print(f"Linking:    vowel={design.boundary_repair.linking_vowel} consonant={design.boundary_repair.linking_consonant}")


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
        if lx.derivation:
            children.setdefault(lx.derivation.base_lexeme_id, []).append(lx)
    roots = sorted((lx for lx in state.lexemes if lx.id in children and not lx.derivation),
                   key=lambda lx: -len(children[lx.id]))

    def show(lx, indent: int) -> None:
        rule = f"  (+{lx.derivation.rule_id})" if lx.derivation else ""
        print(f"{'    ' * indent}{form[lx.id]:<14} {lx.senses[0].gloss}{rule}")
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
    sub.add_parser("graph", help="Print the design workflow as a Mermaid diagram").set_defaults(func=graph_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
