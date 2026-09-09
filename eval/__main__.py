"""``python -m eval`` -- re-render the matrix, examples, and takeaways.

Reads the last run written by the runner (``reports/results.json``) and writes
the README (between markers) and ``reports/matrix.md`` again, without touching
the model. Useful after editing the wording in :mod:`eval.render`.
"""

from __future__ import annotations

from eval.matrix import load_results
from eval.render import render_all


def main() -> int:
    run = load_results()
    for label, path in render_all(run).items():
        print(f"wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
