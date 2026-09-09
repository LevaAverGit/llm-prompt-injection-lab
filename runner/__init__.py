"""Runner package: drives the attack x defense matrix and manages the cache.

This package orchestrates the lab. For every (attack class, defense) cell it
selects the corpus samples, runs each one through the defended toy assistant
under a chosen provider (``ollama`` for the real local model, ``mock`` for a
deterministic offline stand-in), asks the judge whether the defense was
breached, and records the structured :class:`~graph.state.Verdict`. Responses
are cached on disk (keyed by ``provider__model__scenario``) so the whole matrix
reproduces offline.

Scope reminder: everything here exercises the lab's OWN isolated toy assistant
in order to measure and harden its defenses. Nothing targets real, external
systems.
"""

from __future__ import annotations

from .run import CellResult, MatrixRun, run, run_matrix, write_results

__all__ = ["run", "run_matrix", "write_results", "CellResult", "MatrixRun"]
