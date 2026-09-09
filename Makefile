# Prompt-injection resilience lab -- developer entry points.
#
# Defensive, educational lab: everything here exercises the lab's OWN isolated
# toy assistant to measure and harden its defenses. Nothing targets real,
# external systems.

PYTHON ?= python3
VENV   ?= .venv
BIN     = $(VENV)/bin

.PHONY: help install install-guard test run run-llm clean

help:
	@echo "Targets:"
	@echo "  install        Create $(VENV) and install core dependencies"
	@echo "  install-guard  Also install the optional LLM Guard defense (heavy)"
	@echo "  test           Run the parametrized attack regression tests"
	@echo "  run            Rebuild the results matrix from the committed cache (offline)"
	@echo "  run-llm        Re-run the matrix live against Ollama (gemma3:latest)"
	@echo "  clean          Remove caches and the virtualenv"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

install-guard: install
	$(BIN)/pip install -r requirements-guard.txt

test:
	$(BIN)/pytest -q

# Offline: reuse the committed response cache so the matrix reproduces without
# a running model.
run:
	$(BIN)/python -m runner.run_matrix --from-cache

# Live: re-run every attack x defense against the local model and refresh cache.
run-llm:
	$(BIN)/python -m runner.run_matrix --refresh

clean:
	rm -rf $(VENV) .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
