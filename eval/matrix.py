"""Aggregate raw run results into the breach-rate matrix and pick examples.

The runner writes ``reports/results.json`` (metadata + one record per
``(attack sample, defense)`` cell). This module reduces that to:

* a ``attack class x defense`` matrix of **breach rates** (share of the class's
  samples that broke through the defense), assembled with pandas;
* a parallel matrix of ``breached/total`` counts for transparency;
* one worked breach per class -- a real transcript that got through -- so the
  README can show *what* let it in, not only a number.

A cell whose every sample was skipped (e.g. the optional ``llm_guard`` defense
is not installed) is recorded as ``NaN`` / ``"n/a"`` rather than as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from defenses import available_defenses
from graph import AttackClass
from prompts import contains_canary
from runner.run import RESULTS_PATH

# Column and row order for the matrix, taken from the shared enums / registry so
# it never drifts from the rest of the lab.
DEFENSE_ORDER = available_defenses()
CLASS_ORDER = [c.value for c in AttackClass]

# Length of the reply excerpt a reader actually sees in the rendered report. The
# renderer clips each worked-breach reply to this many characters, so the example
# selector below only trusts a canary it can find inside this window. Kept as the
# single source of truth: :mod:`eval.render` imports it for its own clipping.
PREVIEW_CHARS = 1200


def load_results(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the raw run dict written by the runner."""

    target = Path(path) if path is not None else RESULTS_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"no results at {target}. Run `make run` (from cache) or `make run-llm` first."
        )
    return json.loads(target.read_text(encoding="utf-8"))


def _cells(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(run.get("results", []))


def _grid(run: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group cell records by ``[attack_class][defense]``."""

    grid: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        cls: {dfn: [] for dfn in DEFENSE_ORDER} for cls in CLASS_ORDER
    }
    for cell in _cells(run):
        cls = cell.get("attack_class", "")
        dfn = cell.get("defense", "")
        grid.setdefault(cls, {}).setdefault(dfn, []).append(cell)
    return grid


def _columns(run: Dict[str, Any]) -> List[str]:
    """Defense columns actually present in the run, in registry order."""

    seen = {cell.get("defense") for cell in _cells(run)}
    cols = [d for d in DEFENSE_ORDER if d in seen]
    # Preserve any unexpected extra defenses at the end (forward-compatible).
    cols += [d for d in sorted(seen) if d and d not in DEFENSE_ORDER]
    return cols


def _rows(run: Dict[str, Any]) -> List[str]:
    seen = {cell.get("attack_class") for cell in _cells(run)}
    rows = [c for c in CLASS_ORDER if c in seen]
    rows += [c for c in sorted(seen) if c and c not in CLASS_ORDER]
    return rows


def build_rate_matrix(run: Dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame of breach rates (fractions in ``[0, 1]``, ``NaN`` = n/a)."""

    grid = _grid(run)
    rows, cols = _rows(run), _columns(run)
    data: Dict[str, List[float]] = {dfn: [] for dfn in cols}
    for cls in rows:
        for dfn in cols:
            cells = grid.get(cls, {}).get(dfn, [])
            tested = [c for c in cells if c.get("status") == "tested"]
            if not tested:
                data[dfn].append(float("nan"))
            else:
                breached = sum(1 for c in tested if c.get("breached"))
                data[dfn].append(breached / len(tested))
    return pd.DataFrame(data, index=rows, columns=cols)


def build_count_matrix(run: Dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame of ``"breached/total"`` strings (``"n/a"`` when skipped)."""

    grid = _grid(run)
    rows, cols = _rows(run), _columns(run)
    data: Dict[str, List[str]] = {dfn: [] for dfn in cols}
    for cls in rows:
        for dfn in cols:
            cells = grid.get(cls, {}).get(dfn, [])
            tested = [c for c in cells if c.get("status") == "tested"]
            if not tested:
                data[dfn].append("n/a")
            else:
                breached = sum(1 for c in tested if c.get("breached"))
                data[dfn].append(f"{breached}/{len(tested)}")
    return pd.DataFrame(data, index=rows, columns=cols)


def _breaching_turn(cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the first history turn whose verdict reports a breach."""

    for turn in cell.get("history", []):
        verdict = turn.get("verdict") or {}
        if verdict.get("breached"):
            return turn
    return None


def _example_rank(index: int, turn: Dict[str, Any]) -> tuple:
    """Sort key that surfaces the most *verifiable* breach first (lower = better).

    A breach a reviewer can confirm at a glance is a far stronger illustration
    than a generous judge call, so within a class the selector prefers, in order:

    1. a reply whose confidential canary is visible inside the shown excerpt
       (the reader can literally see the leak), then
    2. a reply the judge flagged as a system-prompt leak, then
    3. corpus order, so equally-clear breaches stay stable and reproducible.

    Classes whose breaches are plain instruction-following (no canary, no leak),
    such as ``direct_injection`` and ``role_override``, fall through to corpus
    order and keep their first-by-id example unchanged.
    """

    response = turn.get("response", "") or ""
    verdict = turn.get("verdict") or {}
    canary_visible = contains_canary(response[:PREVIEW_CHARS])
    leaked = bool(verdict.get("leaked_system_prompt"))
    return (0 if canary_visible else 1, 0 if leaked else 1, index)


def select_breach_examples(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pick one worked breach per attack class, with its transcript.

    For each class the weakest defense that was actually breached is preferred
    (so the example is a clear illustration). Within that defense the most
    verifiable breach is chosen via :func:`_example_rank` -- a reply whose leaked
    canary the reader can actually see beats a borderline, generous judge call.
    Classes with no breach anywhere are reported as held.
    """

    grid = _grid(run)
    examples: List[Dict[str, Any]] = []
    for cls in _rows(run):
        chosen: Optional[Dict[str, Any]] = None
        for dfn in _columns(run):
            candidates: List[tuple] = []
            for index, cell in enumerate(grid.get(cls, {}).get(dfn, [])):
                if cell.get("status") == "tested" and cell.get("breached"):
                    turn = _breaching_turn(cell)
                    if turn is not None:
                        candidates.append((_example_rank(index, turn), cell, turn))
            if candidates:
                _, cell, turn = min(candidates, key=lambda item: item[0])
                chosen = {"cell": cell, "turn": turn}
                break
        if chosen is not None:
            cell = chosen["cell"]
            turn = chosen["turn"]
            verdict = turn.get("verdict") or {}
            examples.append(
                {
                    "attack_class": cls,
                    "held": False,
                    "attack_id": cell.get("attack_id", ""),
                    "defense": cell.get("defense", ""),
                    "prompt": turn.get("prompt", ""),
                    "response": turn.get("response", ""),
                    "rationale": verdict.get("rationale", ""),
                    "leaked_system_prompt": bool(verdict.get("leaked_system_prompt")),
                    "attempt": turn.get("attempt", 1),
                }
            )
        else:
            examples.append({"attack_class": cls, "held": True})
    return examples


__all__ = [
    "DEFENSE_ORDER",
    "CLASS_ORDER",
    "PREVIEW_CHARS",
    "load_results",
    "build_rate_matrix",
    "build_count_matrix",
    "select_breach_examples",
]
