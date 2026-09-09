"""The attack corpus loads and every sample has the required fields.

This guards the on-disk corpus (``attacks/*.yml``) and the loader contract: each
file maps to exactly one :class:`AttackClass` (its file stem), every sample
carries ``id`` / ``prompt`` / ``note``, and each stamped sample validates against
the shared :class:`Attack` model with the OWASP ``LLM01`` mapping.
"""

from __future__ import annotations

import os

import pytest

from graph import Attack, AttackClass, OwaspCategory
from tests.support import load_corpus, load_corpus_files

EXPECTED_CLASSES = {cls.value for cls in AttackClass}
CORPUS_FILES = load_corpus_files()
CORPUS = load_corpus()


def test_one_corpus_file_per_attack_class() -> None:
    stems = {os.path.splitext(os.path.basename(path))[0] for path, _ in CORPUS_FILES}
    assert stems == EXPECTED_CLASSES


@pytest.mark.parametrize(
    "path,doc",
    CORPUS_FILES,
    ids=[os.path.basename(path) for path, _ in CORPUS_FILES],
)
def test_corpus_file_structure(path: str, doc: dict) -> None:
    stem = os.path.splitext(os.path.basename(path))[0]

    # Top-level keys required by the loader contract.
    assert doc["attack_class"] == stem, "attack_class must equal the file stem"
    assert doc["attack_class"] in EXPECTED_CLASSES
    assert doc["owasp"] == OwaspCategory.LLM01_PROMPT_INJECTION.value == "LLM01"
    assert isinstance(doc["description"], str) and doc["description"].strip()

    samples = doc["samples"]
    assert isinstance(samples, list) and samples, "samples must be a non-empty list"

    for sample in samples:
        assert isinstance(sample["id"], str) and sample["id"].strip()
        assert isinstance(sample["prompt"], str) and sample["prompt"].strip()
        # note is required in the corpus rows and must be a string.
        assert isinstance(sample["note"], str)
        # The id is conventionally namespaced by its class.
        assert sample["id"].startswith(stem)


def test_corpus_stamps_into_valid_attacks() -> None:
    assert len(CORPUS) == 20, "expected 4 classes x 5 samples"

    per_class: dict[str, int] = {}
    for attack in CORPUS:
        assert isinstance(attack, Attack)
        assert isinstance(attack.attack_class, AttackClass)
        # owasp is a required field, defaulted to LLM01 for every sample.
        assert attack.owasp is OwaspCategory.LLM01_PROMPT_INJECTION
        assert attack.owasp.value == "LLM01"
        assert attack.prompt.strip()
        per_class[attack.attack_class.value] = per_class.get(attack.attack_class.value, 0) + 1

    assert per_class == {cls: 5 for cls in EXPECTED_CLASSES}


def test_corpus_ids_are_unique() -> None:
    ids = [attack.id for attack in CORPUS]
    assert len(ids) == len(set(ids)), "attack ids must be unique across the corpus"
