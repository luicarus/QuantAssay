"""Publish hygiene: this repository is open source, so it must not carry the
author's machine details.

The initial scan found 22 hits across 5 published files (usernames in
`/home/<user>/` paths, absolute Windows paths, WSL mount paths) — none of them a
secret, but all of them leaking the author's identity and private directory
layout, and all of them also broken instructions for anyone else who clones.

This is a *portability* check as much as a privacy one: `D:\\SomeUser\\Work\\...`
is wrong on every other machine.

Three tiers, because the right rule differs per tier:

1. **Published files** (what a clone receives) — strict: no machine paths, no
   credentials, no private directory structure.
2. **Local persistent artifacts** (`docs/engineering-notes.md`, `runs/`) — these
   are gitignored, so machine paths are expected and allowed, but they are still
   scanned for *credentials*: a local file can be screenshotted or pasted.
3. **Ephemeral scratch** (dot-directories created by tooling: `.pytest-tmp`,
   `.scripts`, `__pycache__`, ...) — not scanned. These are throwaway working
   files, and scanning them made this very suite fail depending on which
   platform ran last.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directories whose contents a clone never receives (see .gitignore). Any
#: dot-prefixed directory is treated as tooling scratch by convention, so a new
#: scratch dir does not silently become "published" and break this suite.
IGNORED_DIRS = {
    "runs", ".git", "__pycache__", ".pytest_cache", ".pytest-tmp", ".scan_tmp",
    ".scripts", ".venv", ".ruff_cache", ".mypy_cache",
}
IGNORED_RELPATHS = {
    # Local working material (see .gitignore): everything under docs/ except
    # guide.md, plus all of examples/. These carry this machine's paths and
    # measured-once numbers by design, so the path rule does not apply to them —
    # but the credential rule still does, since a local file can be shared.
    "docs/compatibility.md",
    "docs/engineering-notes.md",
    "docs/plans/full-prd.md",
    "docs/plans/mvp-prd.md",
    "examples/RUNBOOK.md",
    # This file necessarily contains leak-shaped strings as fixtures for the
    # scanner's own test; scanning it would be self-defeating.
    "tests/unit/test_no_local_leaks.py",
}
#: Whole directories that are local-only by .gitignore.
IGNORED_LOCAL_DIRS = {"examples"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".log", ".safetensors"}

#: Local persistent material: machine paths are normal, credentials are not.
LOCAL_ARTIFACT_PREFIXES = ("runs/",)

#: (name, pattern, why it is a problem)
PATH_PATTERNS = [
    ("absolute home path",
     re.compile(r"/home/(?!<)[a-z][a-z0-9_-]{2,}/"),
     "exposes the username and the private home layout"),
    ("absolute Windows path",
     # Both separators: `D:\Work\...` and `D:/Work/...` leak the same username,
     # and the forward-slash form was missed by an earlier version of this rule.
     re.compile(r"\b[A-Za-z]:[\\/](?!<)[A-Za-z0-9_]"),
     "is wrong on every other machine (and carries the username)"),
    ("concrete WSL mount path",
     re.compile(r"/mnt/[a-z]/(?!<)[A-Za-z0-9_]"),
     "bakes in one machine's drive mapping"),
]

#: Private environments/models that belong to the author, not to the project.
#:
#: These were named in an earlier version of the docs ("the author already has a
#: venv called X"). Naming them is both a privacy hint and — more practically —
#: confusing, because a reader has no such environment. Note this is deliberately
#: NOT a general `~/...` ban: `~` expands to the *reader's* home, so
#: `python -m venv ~/venvs/llmcompare` is a portable instruction carrying no
#: username. An earlier version flagged all `~/venvs/` and blocked exactly that
#: correct instruction.
PRIVATE_ENV_PATTERNS = [
    ("author's private venv",
     re.compile(r"~/venvs/(sglang|vllm)(?![\w.-])"),
     "names an environment the project does not own"),
    ("author's private model dir",
     re.compile(r"~/models/Qwen2-VL"),
     "names a model directory that is not part of this project"),
    ("author's private model cache",
     re.compile(r"~/models/llmcompare-cache/hub/models--(?!Qwen--Qwen3-0\.6B)"),
     "names a model outside the one this project measures"),
]

SECRET_PATTERNS = [
    ("HuggingFace token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(r"\bghp_[A-Za-z0-9]{20,}")),
    ("private key material", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("assigned secret", re.compile(
        r"(?i)\b(api[_-]?key|password|passwd|secret)\b\s*[:=]\s*[\"'][A-Za-z0-9_\-]{16,}[\"']")),
]

#: Patterns that look forbidden but are legitimate templates/system paths.
ALLOWED_SUBSTRINGS = [
    "/usr/bin/", "/usr/lib/", "/usr/share/", "/usr/include/",
    "/mnt/<", "$HOME", "<venv>", "<repo>", "<models>", "<drive>", "<path-to>",
    "<40-char-revision>", "<40", "/path/to/",
]


def _safe_is_file(path: Path) -> bool:
    """`is_file()` raises on entries Windows cannot stat (seen in practice).

    A WSL-created symlink left in a scratch directory made `is_file()` raise
    WinError 1920 rather than return False, which crashed the scanner. An entry
    that cannot be inspected is treated as not-a-readable-file.
    """
    try:
        return path.is_file()
    except OSError:
        return False


def _iter_readable_files():
    for path in REPO_ROOT.rglob("*"):
        if not _safe_is_file(path) or path.suffix in IGNORED_SUFFIXES:
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if any(part in IGNORED_LOCAL_DIRS for part in rel.parts):
            continue
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        if rel.as_posix() in IGNORED_RELPATHS:
            continue
        yield path, rel


def _is_local_artifact(rel: Path) -> bool:
    return rel.as_posix().startswith(LOCAL_ARTIFACT_PREFIXES)


def _public_files():
    return [(p, rel) for p, rel in _iter_readable_files() if not _is_local_artifact(rel)]


def _scan_paths(text: str) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(allowed in line for allowed in ALLOWED_SUBSTRINGS):
            continue
        for name, pattern, why in (*PATH_PATTERNS, *PRIVATE_ENV_PATTERNS):
            if pattern.search(line):
                hits.append((lineno, name, why))
                break
    return hits


def _scan_secrets(text: str) -> list[int]:
    hits: list[int] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for _name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                hits.append(lineno)
                break
    return hits


# --------------------------------------------------------------------------
# Published surface
# --------------------------------------------------------------------------


def test_no_published_file_leaks_local_paths() -> None:
    """The repository must stay portable and free of the author's machine details."""
    offenders: list[str] = []
    for path, rel in _public_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, name, why in _scan_paths(text):
            offenders.append(f"{rel}:{lineno}  [{name}] {why}")

    assert offenders == [], (
        "published files contain machine-specific paths; use $HOME / <venv> / "
        "<repo> / <models> placeholders instead:\n  " + "\n  ".join(offenders)
    )


def test_no_credentials_anywhere_readable() -> None:
    """Credentials are never acceptable, even in files that are not committed.

    A local file can be copied, screenshotted or pasted, so this covers the
    local-artifact tier as well as the published one.
    """
    offenders: list[str] = []
    for path, rel in _iter_readable_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno in _scan_secrets(text):
            offenders.append(f"{rel}:{lineno}")
    assert offenders == [], "possible credential material:\n  " + "\n  ".join(offenders)


# --------------------------------------------------------------------------
# Scanner self-checks
# --------------------------------------------------------------------------


def test_the_scanner_itself_detects_a_planted_leak() -> None:
    """Guard against a scanner that passes because it matches nothing."""
    planted = (
        "venv at /home/someuser/venvs/x\n"
        "code at D:\\SomeUser\\Work\\Quantassay\n"
        "mount at /mnt/c/Work/repo\n"
        "the author already has ~/venvs/sglang installed\n"
    )
    hits = _scan_paths(planted)
    assert len(hits) == 4, f"missed a planted leak: {hits}"

    # Portable, reader-relative instructions must NOT be flagged: `~` is the
    # reader's home, so these carry no username and work on any machine.
    clean = (
        "venv at $HOME/venvs/x\n"
        "python -m venv ~/venvs/llmcompare\n"
        "source ~/venvs/llmcompare/bin/activate\n"
        "code at /mnt/<drive>/<path-to>/Quantassay\n"
        "python at /usr/bin/python3.12\n"
        "models under <models>/llmcompare-cache\n"
    )
    assert _scan_paths(clean) == [], f"flagged legitimate placeholders: {_scan_paths(clean)}"


def test_the_original_real_leaks_are_still_detected() -> None:
    """Narrowing a rule must not disarm it.

    These are verbatim fragments of the 22 hits the first scan found. When the
    over-broad `~/venvs/` rule was removed, this check caught that the Windows
    rule also missed the forward-slash form (`D:/Work/...`), which leaks the same
    username. Without this test the scanner could be quietly weakened later.
    """
    original_leaks = [
        "- `/home/someuser/venvs/llmcompare-sglang053` — main environment",
        "| path | `/home/someuser/venvs/llmcompare-sglang053` |",
        "#### archived env: `/home/someuser/venvs/llmcompare`",
        "| code path | `D:\\SomeUser\\Work\\Quantassay` |",
        "reached via `/mnt/d/SomeUser/Work/Quantassay`",
        "| cache root | `/home/someuser/models/llmcompare-cache` |",
        "PY=/home/someuser/venvs/llmcompare-sglang053/bin/python",
        "SNAP=/home/someuser/models/llmcompare-cache/hub/models--Qwen--Qwen3-0.6B",
        "- dev dir `D:/SomeUser/Work/Quantassay` (Windows, Python 3.13)",
    ]
    missed = [line for line in original_leaks if not _scan_paths(line)]
    assert missed == [], (
        "the scanner no longer catches leaks it once caught:\n  " + "\n  ".join(missed)
    )


def test_reader_relative_venv_instructions_are_allowed() -> None:
    """A generic `~/...` install instruction is documentation, not a leak.

    An earlier version of this scanner banned all `~/venvs/`, which blocked the
    correct instruction `python -m venv ~/venvs/llmcompare` — the scanner was
    wrong, not the doc.
    """
    portable = [
        "python -m venv ~/venvs/llmcompare",
        "source ~/venvs/llmcompare/bin/activate",
        "pip install -r requirements/execution-layer.txt",
    ]
    for line in portable:
        assert _scan_paths(line) == [], f"false positive on: {line}"

    # But naming the author's own environments stays flagged, because a reader
    # has no such environment and it hints at the author's setup.
    assert _scan_paths("the author has ~/venvs/vllm already")
    assert _scan_paths("private model at ~/models/Qwen2-VL-2B")


def test_the_secret_scanner_detects_planted_credentials() -> None:
    assert _scan_secrets('api_key = "abcdefghijklmnop1234"')
    assert _scan_secrets("token: hf_abcdefghijklmnopqrstuvwx")
    assert _scan_secrets("-----BEGIN RSA PRIVATE KEY-----")
    assert _scan_secrets("no secrets here, just prose") == []


def test_local_artifacts_are_treated_as_unpublished() -> None:
    """Machine paths in gitignored artifacts are expected; credentials are not.

    `runs/` and the engineering notes legitimately record this machine's paths —
    that is why they are gitignored. They must still be credential-free, and the
    path check must not examine them (it would fail on every real run).
    """
    assert _is_local_artifact(Path("runs/full-pipeline/preflight.json"))
    assert _is_local_artifact(Path("docs/engineering-notes.md")) is False

    local_only = _is_local_artifact(Path("runs/x/y.json"))
    assert local_only is True

    public_rels = [rel.as_posix() for _p, rel in _public_files()]
    assert not any(r.startswith("runs/") for r in public_rels)
    assert "docs/engineering-notes.md" not in public_rels


def test_scratch_directories_are_not_scanned() -> None:
    """Tooling scratch must not determine this suite's result.

    The suite once failed depending on which platform ran last: a WSL run left
    `__pycache__` behind and the Windows scan then read it. Any dot-directory is
    now excluded by convention, so a new scratch dir cannot re-break this.
    """
    scanned = [rel.as_posix() for _p, rel in _iter_readable_files()]
    assert not any(
        part in IGNORED_DIRS for r in scanned for part in Path(r).parts
    ), "a scratch/ignored directory leaked into the scan set"


def test_gitignore_keeps_local_material_out_of_a_clone() -> None:
    """The repo's local-only material must stay ignored."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for required in ("runs/", "docs/engineering-notes.md", ".scripts/", "*.log"):
        assert required in gitignore, (
            f"{required!r} is missing from .gitignore; local evidence would be "
            "committed with absolute paths and machine details"
        )
    # Scratch dirs the suite itself creates must be ignored too.
    for scratch in (".pytest_cache/", ".pytest-tmp/"):
        assert scratch in gitignore, f"{scratch!r} missing from .gitignore"