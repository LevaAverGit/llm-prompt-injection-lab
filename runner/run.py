"""Drive the ``attack class x defense`` matrix and manage the response cache.

This is the harness orchestrator. For every (attack sample, defense) pair it:

1. applies the defense to the probe (which may harden the prompt, transform the
   input, block it before the model, or be unavailable);
2. gets the toy assistant's reply from the chosen provider (cached);
3. asks the judge for a structured :class:`~graph.state.Verdict` (cached);
4. on a non-breach, and while the attempt budget allows, lets the attacker
   produce one same-class variation and tries again.

Each exchange is recorded as a :class:`~graph.state.Turn` inside a
:class:`~graph.state.GraphState`, so the run history uses the same shared
contract as the rest of the lab. Results are written under ``reports/`` and,
for a full sweep, aggregated into the README by :mod:`eval`.

Command line::

    python -m runner.run_matrix --all --provider ollama --refresh
    python -m runner.run_matrix --from-cache            # offline, from cache
    python -m runner.run_matrix --attack-class direct_injection --defense none

Scope reminder: everything here exercises the lab's OWN isolated toy assistant
to measure and harden its defenses. Nothing targets any real, external system.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from defenses import Block, Skipped, available_defenses, get_defense
from graph import Attack, AttackClass, GraphState, Turn, Verdict
from prompts import base_system_prompt

from .cache import CacheMode, ResponseCache, make_key
from .corpus import REPO_ROOT, load_corpus
from .providers import get_provider

REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "results.json"


@dataclass
class CellResult:
    """Outcome of running one attack sample against one defense."""

    attack_id: str
    attack_class: str
    defense: str
    provider: str
    model: str
    status: str  # "tested" | "skipped"
    breached: bool
    leaked_system_prompt: bool
    blocked: bool
    attempts_used: int
    defense_action: str
    skip_reason: str
    state: Optional[GraphState] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "attack_id": self.attack_id,
            "attack_class": self.attack_class,
            "defense": self.defense,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "breached": self.breached,
            "leaked_system_prompt": self.leaked_system_prompt,
            "blocked": self.blocked,
            "attempts_used": self.attempts_used,
            "defense_action": self.defense_action,
            "skip_reason": self.skip_reason,
        }
        if self.state is not None:
            payload["history"] = [turn.model_dump() for turn in self.state.history]
        return payload


@dataclass
class MatrixRun:
    """A full or partial sweep: the cells plus run metadata."""

    cells: List[CellResult] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    mode: str = ""
    max_attempts: int = 1
    generated_at: str = ""
    full_matrix: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "provider": self.provider,
                "model": self.model,
                "mode": self.mode,
                "max_attempts": self.max_attempts,
                "generated_at": self.generated_at,
                "full_matrix": self.full_matrix,
            },
            "results": [cell.to_dict() for cell in self.cells],
        }


# ---------------------------------------------------------------------------
# Single cell
# ---------------------------------------------------------------------------


def _block_verdict(defense_name: str, reason: str) -> Verdict:
    """Deterministic non-breach verdict for input blocked before the model."""

    return Verdict(
        breached=False,
        rationale=(
            f"Input blocked by the {defense_name!r} defense before it reached the "
            f"model ({reason}); a canned refusal was returned, so nothing leaked."
        ),
        leaked_system_prompt=False,
    )


def run(
    attack: Attack,
    defense_name: str,
    *,
    provider: Any,
    cache: ResponseCache,
    max_attempts: int = 1,
) -> CellResult:
    """Run one attack sample against one defense and return the outcome."""

    defend = get_defense(defense_name)
    base_prompt = base_system_prompt()

    state = GraphState(
        attack=attack,
        defense_name=defense_name,
        max_attempts=max_attempts,
    )

    current_attack = attack
    last_response = ""
    defense_action = ""
    blocked = False

    for attempt in range(1, max_attempts + 1):
        state.attempt = attempt

        # Attacker: replay verbatim on attempt 1, vary on later attempts.
        if attempt > 1:
            scenario = f"attacker|{defense_name}|{attack.id}|a{attempt}"
            payload = cache.resolve(
                make_key(provider.name, provider.model, scenario),
                lambda: {
                    "kind": "attacker",
                    "output": provider.attacker_variation(attack, last_response),
                },
            )
            varied_prompt = payload["output"] or attack.prompt
            current_attack = attack.model_copy(update={"prompt": varied_prompt})

        # Defense: harden / transform / block / skip.
        outcome, meta = defend(current_attack.prompt, base_prompt)
        defense_action = str(meta.get("action", ""))

        if isinstance(outcome, Skipped):
            return CellResult(
                attack_id=attack.id,
                attack_class=attack.attack_class.value,
                defense=defense_name,
                provider=provider.name,
                model=provider.model,
                status="skipped",
                breached=False,
                leaked_system_prompt=False,
                blocked=False,
                attempts_used=attempt - 1,
                defense_action="skipped",
                skip_reason=outcome.reason,
                state=state,
            )

        if isinstance(outcome, Block):
            blocked = True
            response = outcome.response
            verdict = _block_verdict(defense_name, outcome.reason)
        else:  # EffectivePrompt -> ask the model
            scenario = f"app|{defense_name}|{attack.id}|a{attempt}"
            app_payload = cache.resolve(
                make_key(provider.name, provider.model, scenario),
                lambda: {
                    "kind": "app",
                    "system_prompt": outcome.system_prompt,
                    "user_input": outcome.user_input,
                    "output": provider.app_reply(outcome.system_prompt, outcome.user_input),
                },
            )
            response = app_payload["output"]

            judge_scenario = f"judge|{defense_name}|{attack.id}|a{attempt}"
            judge_payload = cache.resolve(
                make_key(provider.name, provider.model, judge_scenario),
                lambda: {
                    "kind": "judge",
                    "response": response,
                    "verdict": provider.judge_verdict(current_attack, response).model_dump(),
                },
            )
            verdict = Verdict(**judge_payload["verdict"])

        turn = Turn(
            attempt=attempt,
            prompt=current_attack.prompt,
            response=response,
            verdict=verdict,
        )
        state.history.append(turn)
        state.verdict = verdict
        last_response = response

        if verdict.breached:
            break

    return CellResult(
        attack_id=attack.id,
        attack_class=attack.attack_class.value,
        defense=defense_name,
        provider=provider.name,
        model=provider.model,
        status="tested",
        breached=state.breached,
        leaked_system_prompt=any(
            t.verdict is not None and t.verdict.leaked_system_prompt for t in state.history
        ),
        blocked=blocked,
        attempts_used=len(state.history),
        defense_action=defense_action,
        skip_reason="",
        state=state,
    )


# ---------------------------------------------------------------------------
# Full / partial matrix
# ---------------------------------------------------------------------------


def run_matrix(
    *,
    attack_classes: Optional[List[AttackClass]] = None,
    defenses: Optional[List[str]] = None,
    provider: Any,
    cache: ResponseCache,
    max_attempts: int = 1,
    on_cell: Optional[Any] = None,
) -> MatrixRun:
    """Run every selected (attack sample, defense) pair.

    ``on_cell`` is an optional callback invoked with each :class:`CellResult`
    as it completes, so a caller can stream progress.
    """

    corpus = load_corpus()
    classes = attack_classes or list(AttackClass)
    defense_names = defenses or available_defenses()

    cells: List[CellResult] = []
    for attack_class in classes:
        for attack in corpus[attack_class]:
            for defense_name in defense_names:
                cell = run(
                    attack,
                    defense_name,
                    provider=provider,
                    cache=cache,
                    max_attempts=max_attempts,
                )
                cells.append(cell)
                if on_cell is not None:
                    on_cell(cell)

    full = set(classes) == set(AttackClass) and set(defense_names) == set(available_defenses())
    return MatrixRun(
        cells=cells,
        provider=provider.name,
        model=provider.model,
        mode=cache.mode.value,
        max_attempts=max_attempts,
        generated_at=_dt.datetime.now().replace(microsecond=0).isoformat(),
        full_matrix=full,
    )


def write_results(matrix: MatrixRun, path: Path = RESULTS_PATH) -> Path:
    """Persist the full run (metadata + every cell) to ``reports/results.json``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(matrix.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------


def _print_summary(matrix: MatrixRun) -> None:
    """Print a compact breach-rate table to the terminal (rich if available)."""

    classes = [c.value for c in AttackClass]
    defenses = available_defenses()

    # breaches[class][defense] = list of per-sample breach booleans.
    breaches: Dict[str, Dict[str, List[bool]]] = {c: {d: [] for d in defenses} for c in classes}
    skipped: Dict[str, set] = {c: set() for c in classes}
    for cell in matrix.cells:
        breaches.setdefault(cell.attack_class, {}).setdefault(cell.defense, [])
        if cell.status == "skipped":
            skipped.setdefault(cell.attack_class, set()).add(cell.defense)
            continue
        breaches[cell.attack_class][cell.defense].append(cell.breached)

    def rate(cls: str, dfn: str) -> str:
        vals = breaches.get(cls, {}).get(dfn, [])
        if not vals:
            return "n/a" if dfn in skipped.get(cls, set()) else "-"
        return f"{round(100 * sum(vals) / len(vals))}%"

    present = [cls for cls in classes if any(breaches.get(cls, {}).get(d) for d in defenses)]
    header = ["attack_class \\ defense"] + defenses
    rows = [[cls] + [rate(cls, d) for d in defenses] for cls in present]
    if not rows:
        return

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Breach rate  (provider={matrix.provider}, model={matrix.model})")
        for col in header:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        Console().print(table)
    except Exception:
        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
        print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
        print("  ".join("-" * widths[i] for i in range(len(header))))
        for row in rows:
            print("  ".join(row[i].ljust(widths[i]) for i in range(len(header))))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runner.run_matrix",
        description=(
            "Run the prompt-injection resilience matrix (attack class x defense) "
            "for the lab's own toy assistant."
        ),
    )
    parser.add_argument(
        "--attack-class",
        choices=[c.value for c in AttackClass],
        help="Restrict to a single attack class (default: all).",
    )
    parser.add_argument(
        "--defense",
        choices=available_defenses(),
        help="Restrict to a single defense (default: all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Explicitly run the full matrix (all classes x all defenses). This is "
            "already the default, so --all is only meaningful as a guard: it errors "
            "if combined with --attack-class or --defense."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "mock"],
        default="ollama",
        help="Model provider (default: ollama).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Override the provider's model name.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-cache",
        action="store_true",
        help="Reproduce from the committed cache only; a miss is an error (offline).",
    )
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the cache and call the model, overwriting stored responses.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Attempt budget per sample; >1 enables attacker variations (default: 1).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Do not render the README / matrix report after a full run.",
    )
    return parser


def _select_mode(args: argparse.Namespace) -> CacheMode:
    if args.from_cache:
        return CacheMode.FROM_CACHE
    if args.refresh:
        return CacheMode.REFRESH
    return CacheMode.AUTO


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.all and (args.attack_class or args.defense):
        print(
            "--all runs the full matrix and cannot be combined with "
            "--attack-class/--defense",
            file=sys.stderr,
        )
        return 2

    attack_classes = [AttackClass(args.attack_class)] if args.attack_class else None
    defenses = [args.defense] if args.defense else None

    provider = get_provider(args.provider, args.model)
    cache = ResponseCache(mode=_select_mode(args))

    if args.max_attempts < 1:
        print("--max-attempts must be >= 1", file=sys.stderr)
        return 2

    print(
        f"Running matrix: provider={provider.name} model={provider.model} "
        f"mode={cache.mode.value} max_attempts={args.max_attempts}",
        file=sys.stderr,
    )

    def _progress(cell: CellResult) -> None:
        if cell.status == "skipped":
            flag = "skip"
        elif cell.breached:
            flag = "BREACH"
        else:
            flag = "held"
        print(f"  [{flag:>6}] {cell.attack_id:<24} {cell.defense}", file=sys.stderr)

    matrix = run_matrix(
        attack_classes=attack_classes,
        defenses=defenses,
        provider=provider,
        cache=cache,
        max_attempts=args.max_attempts,
        on_cell=_progress,
    )

    # Only a full sweep owns the canonical results.json that eval / the README
    # read; a partial run is written aside so it never clobbers the full matrix.
    results_path = RESULTS_PATH if matrix.full_matrix else REPORTS_DIR / "results.partial.json"
    write_results(matrix, results_path)
    _print_summary(matrix)

    if matrix.full_matrix and not args.no_render:
        # Imported lazily so a partial/CLI run does not require pandas.
        from eval import render_all

        outputs = render_all(matrix.to_dict())
        for label, path in outputs.items():
            print(f"wrote {label}: {path}", file=sys.stderr)
    elif not matrix.full_matrix:
        print(
            "partial run: skipped README/matrix rendering (use --all for the full sweep)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CellResult",
    "MatrixRun",
    "run",
    "run_matrix",
    "write_results",
    "main",
    "REPORTS_DIR",
    "RESULTS_PATH",
]
