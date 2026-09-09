"""Load the attack corpus from ``attacks/*.yml`` into :class:`Attack` objects.

Each YAML file is one attack class. Its top-level ``attack_class`` and ``owasp``
keys are stamped onto every sample under ``samples``, so a sample row only needs
``id`` / ``prompt`` / ``note``. Every loaded row is validated against the shared
:class:`~graph.state.Attack` contract, so a malformed corpus fails loudly here
rather than deep inside a run.

Scope reminder: the corpus contains generic, illustrative probes aimed only at
the lab's OWN isolated toy assistant. It exists to measure and harden that
single fictional target -- never to attack any real, external system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from graph import Attack, AttackClass, OwaspCategory

# Repository root = the parent of this ``runner`` package.
REPO_ROOT = Path(__file__).resolve().parent.parent
ATTACKS_DIR = REPO_ROOT / "attacks"


class CorpusError(RuntimeError):
    """Raised when a corpus file is missing, malformed, or inconsistent."""


def _load_file(path: Path) -> List[Attack]:
    """Parse a single ``attacks/<class>.yml`` file into a list of attacks."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise CorpusError(f"{path.name}: not valid YAML ({exc})") from exc

    if not isinstance(raw, dict):
        raise CorpusError(f"{path.name}: top level must be a mapping")

    declared_class = raw.get("attack_class")
    if declared_class is None:
        raise CorpusError(f"{path.name}: missing top-level 'attack_class'")

    # The file stem, the declared class, and the enum value must all agree.
    if declared_class != path.stem:
        raise CorpusError(
            f"{path.name}: attack_class {declared_class!r} does not match file "
            f"stem {path.stem!r}"
        )
    try:
        attack_class = AttackClass(declared_class)
    except ValueError as exc:
        raise CorpusError(
            f"{path.name}: {declared_class!r} is not a known AttackClass"
        ) from exc

    owasp_value = raw.get("owasp", OwaspCategory.LLM01_PROMPT_INJECTION.value)
    try:
        owasp = OwaspCategory(owasp_value)
    except ValueError as exc:
        raise CorpusError(f"{path.name}: {owasp_value!r} is not a known OWASP category") from exc

    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise CorpusError(f"{path.name}: 'samples' must be a non-empty list")

    attacks: List[Attack] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise CorpusError(f"{path.name}: sample #{index} is not a mapping")
        # Stamp the file-level class/owasp onto each row, then validate.
        try:
            attack = Attack(
                id=sample.get("id"),
                attack_class=attack_class,
                owasp=owasp,
                prompt=(sample.get("prompt") or "").strip(),
                note=(sample.get("note") or "").strip(),
            )
        except Exception as exc:  # pydantic ValidationError and friends
            ident = sample.get("id", f"#{index}")
            raise CorpusError(f"{path.name}: sample {ident!r} is invalid: {exc}") from exc
        attacks.append(attack)

    return attacks


def load_corpus(attacks_dir: Optional[Path] = None) -> Dict[AttackClass, List[Attack]]:
    """Load every ``attacks/*.yml`` file into ``{AttackClass: [Attack, ...]}``.

    The mapping is ordered by :class:`AttackClass` declaration order so the
    results matrix rows come out in a stable, predictable sequence.
    """

    directory = attacks_dir or ATTACKS_DIR
    if not directory.is_dir():
        raise CorpusError(f"attack corpus directory not found: {directory}")

    by_class: Dict[AttackClass, List[Attack]] = {}
    for attack_class in AttackClass:
        path = directory / f"{attack_class.value}.yml"
        if not path.is_file():
            raise CorpusError(f"missing corpus file for {attack_class.value}: {path}")
        by_class[attack_class] = _load_file(path)

    # Guard against duplicate ids across the whole corpus (ids double as cache
    # keys and pytest ids, so they must be globally unique).
    seen: Dict[str, str] = {}
    for attack_class, attacks in by_class.items():
        for attack in attacks:
            if attack.id in seen:
                raise CorpusError(
                    f"duplicate attack id {attack.id!r} in {attack_class.value} "
                    f"(also in {seen[attack.id]})"
                )
            seen[attack.id] = attack_class.value

    return by_class


def load_attacks(attacks_dir: Optional[Path] = None) -> List[Attack]:
    """Return every attack in the corpus as a flat list (class order preserved)."""

    flat: List[Attack] = []
    for attacks in load_corpus(attacks_dir).values():
        flat.extend(attacks)
    return flat


def load_class(attack_class: AttackClass, attacks_dir: Optional[Path] = None) -> List[Attack]:
    """Return the samples for a single attack class."""

    return load_corpus(attacks_dir)[attack_class]


__all__ = [
    "CorpusError",
    "REPO_ROOT",
    "ATTACKS_DIR",
    "load_corpus",
    "load_attacks",
    "load_class",
]
