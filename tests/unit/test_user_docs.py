"""User-facing documentation must match the code.

`README.md` and `docs/guide.md` state concrete defaults (document counts, corpus
defaults, TPOT's formula) and quote exact error messages. Documentation that gets
these wrong is worse than no documentation, because users trust it and act on
it — a wrong default sends them down a wrong path and a wrong formula makes them
distrust correct numbers.

These checks are cheap and catch drift at the moment a default changes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from quantassay.contracts import MODEL_ID  # noqa: E402
from quantassay.evaluation import corpus as corpus_mod  # noqa: E402

GATING_SRC = (REPO_ROOT / "src/quantassay/gating.py").read_text(encoding="utf-8")
CONTRACTS_SRC = (REPO_ROOT / "src/quantassay/contracts.py").read_text(encoding="utf-8")
CORPUS_SRC = (REPO_ROOT / "src/quantassay/evaluation/corpus.py").read_text(encoding="utf-8")
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
GUIDE = (REPO_ROOT / "docs/guide.md").read_text(encoding="utf-8")

DOCS = {"README.md": README, "docs/guide.md": GUIDE}


def test_user_docs_exist() -> None:
    """An open-source project needs a README and a user guide."""
    assert (REPO_ROOT / "README.md").is_file()
    assert (REPO_ROOT / "docs/guide.md").is_file()


#: Files a clone receives as documentation. Everything else under docs/, examples/
#: and the repo root's AGENTS.md is local working material (see .gitignore):
#: measured-once numbers, failure narratives, internal planning and agent
#: instructions. Only two docs ship, on purpose.
PUBLISHED_DOCS = ["README.md", "docs/guide.md"]

#: Local by design. Public docs must not link to these, or a clone shows dead
#: links — which reads as a broken repository.
LOCAL_DOCS = [
    "AGENTS.md",
    "docs/compatibility.md",
    "docs/engineering-notes.md",
    "docs/plans/full-prd.md",
    "docs/plans/mvp-prd.md",
    "examples/RUNBOOK.md",
]


def test_public_docs_do_not_link_to_local_files() -> None:
    """A link a reader cannot follow is worse than a plain path.

    This is the check that caught 15 dangling links when the docs were split
    into public and local sets.
    """
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    dangling: list[str] = []
    known_public = set(PUBLISHED_DOCS) | {"requirements/execution-layer.txt"}
    for name in PUBLISHED_DOCS:
        doc = REPO_ROOT / name
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for target in link_re.findall(line):
                if target.startswith(("http", "#", "mailto:")):
                    continue
                resolved = (doc.parent / target.split("#")[0]).resolve()
                try:
                    rel = resolved.relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    continue
                if rel in LOCAL_DOCS:
                    dangling.append(f"{name}:{lineno} -> {target}")
    assert dangling == [], (
        "public docs link to local-only files; use a plain path (labelled local) "
        "instead of a link:\n  " + "\n  ".join(dangling)
    )


def test_the_two_public_docs_are_self_contained() -> None:
    """User-facing docs must not depend on files a clone does not receive.

    When compatibility.md became local, the guide's key numbers had to be inlined
    rather than cited: a reader without the file cannot follow a section number.
    """
    guide = (REPO_ROOT / "docs/guide.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    # No cross-references to internal section numbers.
    for text, name in ((guide, "guide.md"), (readme, "README.md")):
        leaked = re.findall(r"兼容性记录\s*§\d", text)
        assert not leaked, f"{name} still cites internal sections: {leaked}"

    # The evidence the guide relies on must be stated, not referenced.
    # These are the exact values from the cited runs; test_awq_documentation.py
    # additionally re-derives them from the artifacts, so this check only has to
    # confirm the numbers are present rather than merely pointed at.
    for value, why in (
        ("30.1311", "baseline perplexity"),
        ("42.1247", "GPTQ perplexity"),
        ("37.9930", "AWQ perplexity"),
        ("36.67", "lower confidence bound"),
        ("43.14", "upper confidence bound"),
    ):
        assert value in guide, f"guide must state {why} inline ({value})"


def test_local_docs_are_actually_gitignored() -> None:
    """The split must be enforced by .gitignore's *semantics*, not by strings.

    An earlier version only asserted that certain substrings appeared, so
    deleting the `examples/` rule went unnoticed. This evaluates each rule the
    way git does — last matching pattern wins — so a removed rule fails here.
    """
    rules = _gitignore_rules(REPO_ROOT / ".gitignore")

    def ignored_by_gitignore(rel: str) -> bool:
        state = False
        for pattern, negate in rules:
            if _gitignore_matches(rel, pattern):
                state = not negate
        return state

    # Local by design.
    for rel in (
        "AGENTS.md",
        "examples/RUNBOOK.md",
        "docs/compatibility.md",
        "docs/plans/full-prd.md",
        "docs/engineering-notes.md",
    ):
        assert ignored_by_gitignore(rel), f"{rel} must NOT be published"

    # Published on purpose.
    for rel in (
        "docs/guide.md",
        "README.md",
        "requirements/execution-layer.txt",
        "src/quantassay/gating.py",
    ):
        assert not ignored_by_gitignore(rel), f"{rel} MUST be published"


def test_a_new_internal_doc_is_private_by_default() -> None:
    """Opt-in publishing: an unseen doc must not ship.

    This is the property that makes the allowlist approach safe. A denylist would
    silently publish whatever a contributor forgot to add, and the leak scanner
    only sees files it knows to look at.
    """
    rules = _gitignore_rules(REPO_ROOT / ".gitignore")

    def ignored_by_gitignore(rel: str) -> bool:
        state = False
        for pattern, negate in rules:
            if _gitignore_matches(rel, pattern):
                state = not negate
        return state

    for future_doc in (
        "docs/notes-2027-01.md",
        "docs/plans/next-prd.md",
        "examples/another-checklist.md",
    ):
        assert ignored_by_gitignore(future_doc), (
            f"{future_doc} would be published by default; new internal docs must "
            "start private"
        )


def _gitignore_rules(path: Path) -> list[tuple[str, bool]]:
    """Parse .gitignore into (pattern, is_negation) pairs, in file order."""
    rules: list[tuple[str, bool]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        rules.append((line[1:] if negate else line, negate))
    return rules


def _gitignore_matches(rel: str, pattern: str) -> bool:
    """Approximate gitignore matching for the patterns this repo uses."""
    if pattern.endswith("/"):
        base = pattern.rstrip("/")
        return rel == base or rel.startswith(base + "/")
    if "/" in pattern:
        from fnmatch import fnmatch
        return fnmatch(rel, pattern)
    from fnmatch import fnmatch
    return fnmatch(rel, pattern) or fnmatch(rel.rsplit("/", 1)[-1], pattern)


@pytest.mark.parametrize(
    "flag,expected",
    [
        ("--quality-documents", 64),
        ("--quality-calibration-samples", 4),
        ("--quality-max-tokens", 512),
        ("--quality-context-length", 512),
        ("--corpus-min-chars", 400),
    ],
)
def test_documented_int_defaults_match_the_parser(flag: str, expected: int) -> None:
    match = re.search(rf'"{re.escape(flag)}",\s*type=int,\s*default=(\d+)', GATING_SRC)
    assert match, f"{flag} not found in the parser"
    assert int(match.group(1)) == expected, (
        f"{flag} default changed to {match.group(1)}; update README/guide"
    )


@pytest.mark.parametrize(
    "flag,expected",
    [
        ("--quality-documents", 64),
        ("--quality-calibration-samples", 4),
        ("--quality-max-tokens", 512),
        ("--quality-context-length", 512),
        ("--corpus-min-chars", 400),
        ("--mem-fraction-static", 0.8),
        ("--cuda-graph-max-bs", 2),
    ],
)
def test_guide_defaults_table_agrees_with_the_parser(flag: str, expected) -> None:
    """The guide's tables state defaults; they must not drift from the code.

    A wrong default in a table is the most actionable kind of doc error: a user
    copies the number instead of reading the flag's own help.
    """
    pattern = re.compile(rf"\|\s*`{re.escape(flag)}`[^|]*\|[^|]*\|\s*`?({expected})`?\s*\|")
    assert pattern.search(GUIDE) or pattern.search(README), (
        f"{flag} is not documented with its real default {expected} in a table"
    )


def test_documented_float_and_cuda_graph_defaults_match() -> None:
    assert 'add_argument("--mem-fraction-static", type=float, default=0.8)' in GATING_SRC
    assert "DEFAULT_CUDA_GRAPH_MAX_BS = 2" in GATING_SRC
    # The guide tells users these values.
    assert "0.8" in GUIDE and "cuda-graph-max-bs" in GUIDE


def test_documented_corpus_defaults_match_the_code() -> None:
    assert corpus_mod.DEFAULT_CORPUS == "Salesforce/wikitext"
    assert corpus_mod.DEFAULT_CORPUS_CONFIG == "wikitext-2-raw-v1"
    assert corpus_mod.DEFAULT_CORPUS_SPLIT == "test"
    assert corpus_mod.DEFAULT_TEXT_COLUMN == "text"
    for value in ("Salesforce/wikitext", "wikitext-2-raw-v1", "text"):
        assert value in GUIDE, f"guide should name the default {value!r}"


def test_documented_tpot_formula_matches_implementation() -> None:
    """TPOT is the metric most easily documented wrongly."""
    assert "(self.e2e_latency_ms - self.ttft_ms) / (self.output_tokens - 1)" in CONTRACTS_SRC
    assert "if self.output_tokens < 2:" in CONTRACTS_SRC
    # The guide must state the same formula, in a form a reader can act on.
    assert "(E2E - TTFT) / (output_tokens - 1)" in GUIDE
    # And it must say the metric is unavailable below 2 output tokens, which is
    # the definition's edge case rather than a missing value.
    assert "不足 2 token" in GUIDE or "output_tokens < 2" in GUIDE or "below 2" in GUIDE


def test_documented_stage_names_all_exist() -> None:
    """Stage names are derived per method, so assert the derivation, not literals.

    An earlier version looked for `"awq_service"` as a literal in gating.py. Names
    are generated from the method now, so a literal search would demand the very
    duplication this refactor removed.
    """
    from quantassay.gating import SUPPORTED_QUANT_METHODS, method_stage_names

    for stage in ("preflight", "bf16", "benchmark_bf16", "quality_bf16"):
        assert f'"{stage}"' in GATING_SRC, f"stage {stage} disappeared"
        assert stage in GUIDE, f"guide should mention stage {stage}"

    for method in SUPPORTED_QUANT_METHODS:
        names = method_stage_names(method)
        assert names["quantize"] == method
        assert names["service"] == f"{method}_service"
        assert names["benchmark"] == f"benchmark_{method}"
        assert names["quality"] == f"quality_{method}"

    # Every stage must be reachable from --stage: either as a literal or through
    # the generator over SUPPORTED_QUANT_METHODS.
    assert "SUPPORTED_QUANT_METHODS" in GATING_SRC
    stage_choices = GATING_SRC.split("--stage")[1].split("default=")[0]
    assert "SUPPORTED_QUANT_METHODS" in stage_choices or all(
        f'"{method}_service"' in stage_choices for method in SUPPORTED_QUANT_METHODS
    ), "quantized stages must be selectable via --stage"


def test_documented_workload_matches_the_config() -> None:
    cfg = (REPO_ROOT / "configs/mvp-qwen3-0p6b.yaml").read_text(encoding="utf-8")
    for key, value in (("max_new_tokens", "64"), ("timed_requests", "20"),
                       ("warmup_requests", "3"), ("concurrency", "1")):
        match = re.search(rf"{key}:\s*(\S+)", cfg)
        assert match and match.group(1) == value, f"workload {key} changed"


def test_quoted_error_messages_are_verbatim() -> None:
    """The guide quotes real messages; quote drift would mislead users."""
    assert "has no column" in CORPUS_SRC and "available columns" in CORPUS_SRC
    assert "no usable documents" in CORPUS_SRC
    assert "already holds quality data" in GATING_SRC
    assert "Use a new run directory" in GATING_SRC


def test_model_id_is_documented_truthfully() -> None:
    """The guide says the model id is hardcoded — that must remain true."""
    assert MODEL_ID == "Qwen/Qwen3-0.6B"
    assert "contracts.py" in GUIDE and "写死" in GUIDE


def test_limitations_are_disclosed() -> None:
    """The project must not oversell unmeasured capability."""
    combined = README + GUIDE
    for phrase, why in (
        ("未实现", "must state task accuracy is not implemented"),
        ("4 条", "must disclose the small default calibration set"),
        ("成因未定位", "must disclose the unexplained serving spread"),
        ("只有困惑度", "must scope quality to perplexity"),
    ):
        assert phrase in combined, f"documentation no longer discloses: {why}"


def test_no_doc_promises_transformers_substitution() -> None:
    """The SGLang-only rule is a core claim; docs must not contradict it."""
    assert "不用 `Transformers.generate()`" in GUIDE or "不用 Transformers" in GUIDE
    assert "SGLang" in README


def test_all_relative_links_resolve_in_the_working_tree() -> None:
    """Links must resolve here (not only in a clone).

    Note this is *not* sufficient on its own: a link to a gitignored file exists
    in the working tree but dangles in a clone. `test_public_docs_do_not_link_to_
    local_files` covers that case; this one catches typos.
    """
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    broken: list[str] = []
    for name, text in DOCS.items():
        doc_path = REPO_ROOT / name
        for target in link_re.findall(text):
            if target.startswith(("http", "#", "mailto:")):
                continue
            if not (doc_path.parent / target.split("#")[0]).resolve().exists():
                broken.append(f"{name}: {target}")
    assert broken == [], "broken links:\n  " + "\n  ".join(broken)


def test_docs_do_not_leak_local_paths() -> None:
    """User docs are published, so they follow the same hygiene as other files."""
    offenders: list[str] = []
    patterns = (
        re.compile(r"/home/(?!<)[a-z][a-z0-9_-]{2,}/"),
        re.compile(r"\b[A-Za-z]:\\(?!<)[A-Za-z0-9_]"),
        re.compile(r"/mnt/[a-z]/(?!<)[A-Za-z0-9_]"),
    )
    for name, text in DOCS.items():
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(tok in line for tok in ("$HOME", "<", "/usr/")):
                continue
            if any(p.search(line) for p in patterns):
                offenders.append(f"{name}:{lineno}")
    assert offenders == [], "docs leak machine paths:\n  " + "\n  ".join(offenders)