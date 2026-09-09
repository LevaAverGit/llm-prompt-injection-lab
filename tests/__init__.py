"""Test package for the prompt-injection resilience lab.

Marking ``tests`` as a package keeps the test module names unique and, under
pytest's default import mode, puts the repository root on ``sys.path`` so the
suite can ``import graph``, ``import defenses`` and ``import prompts`` while also
importing the shared helpers in :mod:`tests.support`.

Everything under here exercises the lab's OWN isolated toy assistant to measure
and harden its defenses. Nothing targets any real, external system.
"""
