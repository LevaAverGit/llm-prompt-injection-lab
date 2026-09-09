"""Evaluation package: aggregate raw run results into the resilience matrix.

:mod:`eval.matrix` turns the per-cell results produced by :mod:`runner` into a
pandas ``attack class x defense`` breach-rate matrix and picks one worked breach
per class. :mod:`eval.render` turns those into Markdown and writes them into the
README (between markers) and into ``reports/``.

Scope reminder: the numbers describe the resilience of the lab's OWN isolated
toy assistant. They exist to measure and harden that single fictional contour.
"""

from __future__ import annotations

from eval.matrix import (
    build_count_matrix,
    build_rate_matrix,
    load_results,
    select_breach_examples,
)
from eval.render import render_all

__all__ = [
    "load_results",
    "build_rate_matrix",
    "build_count_matrix",
    "select_breach_examples",
    "render_all",
]
