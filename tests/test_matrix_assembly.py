"""The results matrix aggregates verdicts into per-cell breach statistics.

The matrix is ``matrix[defense][attack_class] -> stats``. This checks the
aggregation math on a small toy set (including a skipped, N/A cell) and then on
the real ``none`` vs ``both`` runs over the corpus, which must show a fully
breached baseline and a fully held hardened contour.
"""

from __future__ import annotations

from graph import AttackClass, Verdict
from tests.support import Cell, assemble_matrix, load_corpus, run_attack

CORPUS = load_corpus()


def _breach(leaked: bool = False) -> Verdict:
    return Verdict(breached=True, rationale="breached", leaked_system_prompt=leaked)


def _hold() -> Verdict:
    return Verdict(breached=False, rationale="held", leaked_system_prompt=False)


def test_matrix_math_on_toy_set() -> None:
    di = AttackClass.DIRECT_INJECTION.value
    ro = AttackClass.ROLE_OVERRIDE.value
    cells = [
        # 'none': direct injection breached 2/2, role override breached 1/2.
        Cell(di, "none", _breach(leaked=True)),
        Cell(di, "none", _breach()),
        Cell(ro, "none", _breach()),
        Cell(ro, "none", _hold()),
        # 'both': everything held.
        Cell(di, "both", _hold()),
        Cell(di, "both", _hold()),
        Cell(ro, "both", _hold()),
        Cell(ro, "both", _hold()),
        # 'llm_guard': unavailable -> skipped, must be N/A (verdict is None).
        Cell(di, "llm_guard", None),
        Cell(di, "llm_guard", None),
    ]
    matrix = assemble_matrix(cells)

    assert matrix["none"][di]["rate"] == 1.0
    assert matrix["none"][di]["breached"] == 2
    assert matrix["none"][di]["leaked"] == 1
    assert matrix["none"][ro]["rate"] == 0.5
    assert matrix["none"][ro]["scored"] == 2

    assert matrix["both"][di]["rate"] == 0.0
    assert matrix["both"][ro]["rate"] == 0.0
    assert matrix["both"][di]["breached"] == 0

    # A fully-skipped cell is N/A: nothing scored, rate is None.
    guard = matrix["llm_guard"][di]
    assert guard["total"] == 2
    assert guard["skipped"] == 2
    assert guard["scored"] == 0
    assert guard["rate"] is None


def test_matrix_counts_partial_skips() -> None:
    di = AttackClass.DIRECT_INJECTION.value
    cells = [
        Cell(di, "mixed", _breach()),
        Cell(di, "mixed", None),  # one skipped
        Cell(di, "mixed", _hold()),
    ]
    matrix = assemble_matrix(cells)
    stats = matrix["mixed"][di]
    assert stats["total"] == 3
    assert stats["skipped"] == 1
    assert stats["scored"] == 2
    assert stats["breached"] == 1
    assert stats["rate"] == 0.5


def test_matrix_from_real_runs_baseline_vs_hardened() -> None:
    """Assemble the matrix from real corpus runs: none = 100%, both = 0%."""
    cells = []
    for attack in CORPUS:
        for defense in ("none", "both"):
            cells.append(Cell(attack.attack_class, defense, run_attack(defense, attack)))
    matrix = assemble_matrix(cells)

    for attack_class in {a.attack_class.value for a in CORPUS}:
        assert matrix["none"][attack_class]["rate"] == 1.0
        assert matrix["both"][attack_class]["rate"] == 0.0

    # Every attack class must be represented on both axes.
    expected = {cls.value for cls in AttackClass}
    assert set(matrix["none"]) == expected
    assert set(matrix["both"]) == expected
