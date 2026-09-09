"""Module entry point: ``python -m runner.run_matrix``.

The Makefile targets ``run`` (``--from-cache``) and ``run-llm`` (``--refresh``)
invoke this module. It is a thin wrapper around :func:`runner.run.main`, kept as
its own module so the command line in the Makefile and README stays stable.
"""

from __future__ import annotations

from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
