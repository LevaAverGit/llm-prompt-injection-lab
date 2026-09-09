"""Render the aggregated results into Markdown and into the README.

Produces three blocks -- the breach-rate matrix, one worked breach per class,
and a data-driven "What I learned" -- and writes them:

* into ``README.md`` between HTML-comment markers (``<!-- BEGIN:MATRIX -->`` and
  so on), so the published README always reflects the latest run;
* into ``reports/matrix.md`` as a standalone report.

Everything describes the resilience of the lab's OWN isolated toy assistant.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from eval.matrix import (
    PREVIEW_CHARS,
    build_count_matrix,
    build_rate_matrix,
    select_breach_examples,
)
from prompts import CANARY
from runner.corpus import REPO_ROOT

README_PATH = REPO_ROOT / "README.md"
REPORTS_DIR = REPO_ROOT / "reports"
MATRIX_REPORT_PATH = REPORTS_DIR / "matrix.md"

# How much of a worked-breach reply to show. Shared with the example selector in
# :mod:`eval.matrix` so "the canary is visible in the excerpt" means the same
# thing there and here.
_MAX_RESPONSE_CHARS = PREVIEW_CHARS

# Sentinel the deterministic canary backstop leaves in a rationale when it had to
# override the LLM judge. If it is present the judge's own narrative may say the
# opposite ("refused", "not leaked"), so the report substitutes a clean line.
_BACKSTOP_MARKER = "(Canary marker detected"


# ---------------------------------------------------------------------------
# Cell / table formatting
# ---------------------------------------------------------------------------


def _rate_cell(rate: float, count: str) -> str:
    if isinstance(rate, float) and math.isnan(rate):
        return "n/a"
    return f"{round(100 * rate)}% ({count})"


def render_matrix_markdown(run: Dict[str, Any]) -> str:
    """Render the breach-rate matrix as a Markdown table plus a provenance line."""

    rates = build_rate_matrix(run)
    counts = build_count_matrix(run)

    cols = list(rates.columns)
    header = "| Attack class \\ Defense | " + " | ".join(f"`{c}`" for c in cols) + " |"
    sep = "|" + "---|" + "|".join([":---:"] * len(cols)) + "|"

    lines = [header, sep]
    for cls in rates.index:
        cells = [
            _rate_cell(rates.loc[cls, c], str(counts.loc[cls, c])) for c in cols
        ]
        lines.append(f"| `{cls}` | " + " | ".join(cells) + " |")

    meta = run.get("meta", {})
    per_class = _samples_per_class(run)
    budget = meta.get("max_attempts", 1)
    try:
        single_attempt = int(budget) <= 1
    except (TypeError, ValueError):
        single_attempt = False
    loop_note = ""
    if single_attempt:
        loop_note = (
            "The attacker → app → judge harness supports a bounded multi-attempt "
            "retry loop (the attacker rephrases a probe that did not break through "
            "and tries again); this shipped run uses a single attempt for a stable, "
            "reproducible matrix. "
        )
    provenance = (
        f"Cells show the **breach rate** — the share of a class's samples that broke "
        f"through the defense (`breached/total` in parentheses). **Lower is better.** "
        f"Measured with provider `{meta.get('provider', '?')}`, model "
        f"`{meta.get('model', '?')}`, {per_class} samples per class, attempt budget "
        f"{budget}. {loop_note}`n/a` = defense not available in this "
        f"environment (e.g. the optional `llm_guard` scanner was not installed). "
        f"Generated from `reports/results.json` ({meta.get('generated_at', '')})."
    )
    return provenance + "\n\n" + "\n".join(lines)


def _samples_per_class(run: Dict[str, Any]) -> int:
    counts = {}
    for cell in run.get("results", []):
        key = (cell.get("attack_class"), cell.get("defense"))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def _fence(text: str) -> str:
    """Wrap text in a code fence long enough not to collide with its content."""

    longest = 0
    for match in re.finditer(r"`+", text):
        longest = max(longest, len(match.group(0)))
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}\n{text}\n{ticks}"


def _clip(text: str) -> str:
    text = (text or "").strip()
    if len(text) > _MAX_RESPONSE_CHARS:
        return text[:_MAX_RESPONSE_CHARS].rstrip() + "\n… [truncated]"
    return text


def _display_rationale(ex: Dict[str, Any]) -> str:
    """Rationale to print for a worked breach, kept honest about the backstop.

    When the deterministic canary backstop overrode the LLM judge, the stored
    rationale carries :data:`_BACKSTOP_MARKER` and may open with a now-wrong
    judge narrative (e.g. "refused to comply"). In that case a self-authored line
    that names the exposed canary is shown instead, so the "why it's a breach"
    text never contradicts the reply above it.
    """

    rationale = (ex.get("rationale") or "").strip()
    if ex.get("leaked_system_prompt") and _BACKSTOP_MARKER in rationale:
        return (
            f"The reply exposed the confidential canary `{CANARY}`, so the "
            "deterministic backstop records it as a system-prompt leak — a breach "
            "regardless of the judge's own narrative."
        )
    return rationale


def render_examples_markdown(run: Dict[str, Any]) -> str:
    """Render one worked breach per class (or a note that the class held)."""

    examples = select_breach_examples(run)
    blocks: List[str] = [
        "One worked breach per attack class, taken straight from the run "
        "transcript, with the exact reply and why the judge scored it a breach. "
        "Where a class was **not** breached by any defense, that is stated too — "
        "a defense holding is a first-class result here."
    ]
    for ex in examples:
        cls = ex["attack_class"]
        if ex.get("held"):
            blocks.append(
                f"### `{cls}` — held\n\n"
                f"No sample in this class broke through any defense in this run."
            )
            continue
        leak_note = " (system-prompt leak)" if ex.get("leaked_system_prompt") else ""
        blocks.append(
            f"### `{cls}` — breached `{ex['defense']}`{leak_note}\n\n"
            f"**Probe** (`{ex['attack_id']}`, attempt {ex.get('attempt', 1)}):\n\n"
            f"{_fence(_clip(ex['prompt']))}\n\n"
            f"**Assistant reply:**\n\n"
            f"{_fence(_clip(ex['response']))}\n\n"
            f"**Why it counts as a breach:** {_display_rationale(ex)}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# What I learned (data-driven)
# ---------------------------------------------------------------------------


def _col_mean(rates: pd.DataFrame, col: str) -> float:
    if col not in rates.columns:
        return float("nan")
    series = rates[col].dropna()
    return float(series.mean()) if not series.empty else float("nan")


def _pct(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{round(100 * value)}%"


def render_learned_markdown(run: Dict[str, Any]) -> str:
    """Generate honest takeaways computed from the matrix."""

    rates = build_rate_matrix(run)
    bullets: List[str] = []

    if "none" in rates.columns:
        none_mean = _col_mean(rates, "none")
        bullets.append(
            f"- **The naive baseline is porous.** With no defense (`none`), the mean "
            f"breach rate across classes is **{_pct(none_mean)}** — the un-hardened "
            f"FAQ prompt gives up its role and its canary readily."
        )

    # Which non-baseline defenses drive every class to zero?
    stoppers: List[str] = []
    partial: List[str] = []
    for col in rates.columns:
        if col == "none":
            continue
        series = rates[col].dropna()
        if series.empty:
            continue
        if (series == 0).all():
            stoppers.append(col)
        elif series.max() < 1.0 and float(series.mean()) < _col_mean(rates, "none"):
            partial.append(col)

    if stoppers:
        names = ", ".join(f"`{s}`" for s in stoppers)
        bullets.append(
            f"- **What fully held.** {names} stopped every probe in every class in "
            f"this run (0% breach). Explicit delimiting, spotlighting untrusted input, "
            f"and a decode-before-match filter go a long way against illustrative probes."
        )
    if partial:
        names = ", ".join(f"`{p}`" for p in partial)
        bullets.append(
            f"- **What helped but leaked.** {names} cut the breach rate well below the "
            f"baseline yet did **not** reach zero on every class — a reminder that a "
            f"single mitigation is rarely airtight."
        )

    # Where does the strongest deployed defense still break?
    candidate = None
    for col in ("both", "system_prompt", "input_filter"):
        if col in rates.columns and not rates[col].dropna().empty:
            candidate = col
            break
    if candidate is not None:
        weak = [cls for cls in rates.index if _cell_breached(rates, cls, candidate)]
        if weak:
            names = ", ".join(f"`{c}`" for c in weak)
            bullets.append(
                f"- **Where it still bends.** Even under `{candidate}`, {names} was not "
                f"fully shut down — the honest negative result the matrix is meant to surface."
            )
        else:
            bullets.append(
                f"- **Strongest deployed contour.** `{candidate}` held every class to 0% "
                f"in this run; the lab exists to catch the day that regresses."
            )

    # Note the optional scanner if it was skipped everywhere.
    if "llm_guard" in rates.columns and rates["llm_guard"].dropna().empty:
        bullets.append(
            "- **`llm_guard` was not evaluated here** (optional dependency not installed), "
            "so its column is `n/a`. Run `make install-guard` to include the real "
            "industrial scanner in the comparison."
        )

    # Be explicit that the verdict is an LLM-as-judge call, not ground truth: some
    # encoding-class breaches are generous. The deterministic canary backstop is
    # what makes the clear-cut leaks trustworthy; the softer calls are a finding.
    if _cell_breached(rates, "encoding_obfuscation", "none"):
        bullets.append(
            "- **Judge noise is part of the result.** Verdicts come from an "
            "LLM-as-judge, so a few `encoding_obfuscation` breaches under `none` are "
            "generous — the model mangled the payload yet was still scored a breach. "
            "The deterministic canary backstop anchors the unambiguous leaks; the "
            "softer calls are why the run ships with a full transcript you can "
            "re-judge, not just a headline number."
        )

    bullets.append(
        "- **Caveat.** These are numbers for a small local model against a deliberately "
        "illustrative corpus; a larger model, or a cleverer probe, will move them. The "
        "value is the reproducible method, not a claim of absolute safety."
    )
    return "\n".join(bullets)


def _cell_breached(rates: pd.DataFrame, cls: str, col: str) -> bool:
    value = rates.loc[cls, col] if col in rates.columns and cls in rates.index else float("nan")
    if isinstance(value, float) and math.isnan(value):
        return False
    return value > 0


# ---------------------------------------------------------------------------
# README patching
# ---------------------------------------------------------------------------


def _replace_marked(text: str, name: str, content: str) -> str:
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end), re.DOTALL
    )
    if not pattern.search(text):
        raise ValueError(
            f"README is missing the marker pair {begin} ... {end}; add it so the "
            f"renderer knows where to write the {name} block."
        )
    return pattern.sub(f"{begin}\n{content}\n{end}", text)


def patch_readme(run: Dict[str, Any], readme_path: Path = README_PATH) -> Path:
    """Write the matrix, examples, and takeaways into the README between markers."""

    text = readme_path.read_text(encoding="utf-8")
    text = _replace_marked(text, "MATRIX", render_matrix_markdown(run))
    text = _replace_marked(text, "BREACHES", render_examples_markdown(run))
    text = _replace_marked(text, "LEARNED", render_learned_markdown(run))
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def write_matrix_report(run: Dict[str, Any], path: Path = MATRIX_REPORT_PATH) -> Path:
    """Write a standalone Markdown report of the matrix and examples."""

    meta = run.get("meta", {})
    parts = [
        "# Prompt-injection resilience matrix",
        "",
        "Defensive, educational report for the lab's own isolated toy assistant.",
        "",
        "## Breach-rate matrix",
        "",
        render_matrix_markdown(run),
        "",
        "## Worked breaches",
        "",
        render_examples_markdown(run),
        "",
        "## Takeaways",
        "",
        render_learned_markdown(run),
        "",
        f"_Source: reports/results.json — provider `{meta.get('provider','?')}`, "
        f"model `{meta.get('model','?')}`, generated {meta.get('generated_at','')}._",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def render_all(run: Dict[str, Any]) -> Dict[str, Path]:
    """Render every output and return ``{label: path}`` for what was written."""

    outputs: Dict[str, Path] = {}
    outputs["matrix report"] = write_matrix_report(run)
    outputs["README"] = patch_readme(run)
    return outputs


def _main() -> int:
    """``python -m eval.render`` — re-render from the last saved results."""

    from eval.matrix import load_results

    run = load_results()
    for label, path in render_all(run).items():
        print(f"wrote {label}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "README_PATH",
    "MATRIX_REPORT_PATH",
    "render_matrix_markdown",
    "render_examples_markdown",
    "render_learned_markdown",
    "patch_readme",
    "write_matrix_report",
    "render_all",
]
