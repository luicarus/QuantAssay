"""The project's license must stay correct and self-consistent.

Licensing errors are unusually costly: a truncated license text, a leftover
"Proprietary" string, or metadata that contradicts LICENSE are all the kind of
mistake discovered by a downstream user rather than by CI.

The canonical-text check needs a reference copy, which comes from an installed
dependency; when that is unavailable (a clone on Windows) the test falls back to
structural checks that hold anywhere.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LICENSE = REPO_ROOT / "LICENSE"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Markers any faithful Apache-2.0 copy contains. A truncated or hand-typed
#: license usually fails at least one of these.
APACHE_MARKERS = (
    "Apache License",
    "Version 2.0, January 2004",
    "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
    "1. Definitions.",
    "4. Redistribution.",
    "7. Disclaimer of Warranty.",
    "9. Accepting Warranty or Additional Liability.",
    "END OF TERMS AND CONDITIONS",
)

#: Copyright holders who must NOT appear: this license is ours, and shipping
#: another project's notice inside it would be wrong.
FOREIGN_HOLDERS = (
    "SGLang Team",
    "The Hugging Face team",
    "Copyright 2018",
    "Copyright 2023",
    "Meta Platforms",
)


def test_license_file_exists_and_is_complete() -> None:
    assert LICENSE.is_file(), "LICENSE must be present at the repository root"
    text = LICENSE.read_text(encoding="utf-8")
    missing = [m for m in APACHE_MARKERS if m not in text]
    assert missing == [], f"LICENSE is missing Apache-2.0 markers: {missing}"
    # The canonical text is ~11k chars / ~200 lines; anything far smaller is a
    # truncation, and anything far larger probably carries extra notices.
    assert 10_000 < len(text) < 13_000, f"unexpected LICENSE size: {len(text)}"
    assert 150 < len(text.splitlines()) < 260


def test_license_names_no_other_copyright_holder() -> None:
    text = LICENSE.read_text(encoding="utf-8")
    present = [h for h in FOREIGN_HOLDERS if h in text]
    assert present == [], (
        f"LICENSE carries another project's copyright notice: {present}; the "
        "canonical Apache text names only the holder who applies the license"
    )


def test_license_appendix_names_the_copyright_holder() -> None:
    """The appendix boilerplate must carry a real holder, not placeholders.

    Apache's distributed text leaves `[yyyy] [name of copyright owner]` for the
    project owner to fill in. Shipping the placeholders unchanged would mean the
    appendix notice is unusable as a copyright statement.
    """
    text = LICENSE.read_text(encoding="utf-8")
    assert "APPENDIX: How to apply the Apache License" in text

    # The placeholders may only remain inside the *instructions* paragraph,
    # which explicitly tells the reader to replace them. The boilerplate itself
    # must not still contain them.
    appendix = text[text.find("APPENDIX: How to apply the Apache License"):]
    boilerplate = appendix[appendix.find("Copyright"):]
    assert "[yyyy]" not in boilerplate, (
        "the appendix boilerplate still has the year placeholder; years are"
    )
    assert "[name of copyright owner]" not in boilerplate, (
        "the appendix boilerplate still has the copyright-holder placeholder"
    )

    # A dated copyright line with a named holder must be present.
    assert re.search(r"Copyright\s+\d{4}\s+\S", boilerplate), (
        "the appendix must name a copyright year and holder"
    )


def test_license_uses_lf_line_endings() -> None:
    raw = LICENSE.read_bytes()
    assert b"\r" not in raw, "LICENSE must use LF; CRLF breaks shell tooling"
    assert raw.endswith(b"\n"), "LICENSE should end with a newline"


def test_pyproject_declares_apache_spdx() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    assert project["license"] == "Apache-2.0", (
        "license must be the PEP 639 SPDX string; the deprecated table form "
        "cannot express a simple SPDX id"
    )
    assert not isinstance(project["license"], dict)
    assert "LICENSE" in project.get("license-files", []), (
        "the LICENSE file must be declared so builds ship it"
    )


def test_build_requirement_supports_pep639() -> None:
    """The SPDX string form only parses on setuptools >= 77.

    Declaring `license = "Apache-2.0"` while allowing an older backend would make
    a fresh `pip install` fail at metadata generation.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requires = data["build-system"]["requires"]
    floors = []
    for req in requires:
        m = re.match(r"setuptools\s*>=\s*(\d+)", req)
        if m:
            floors.append(int(m.group(1)))
    assert floors, f"no setuptools floor declared: {requires}"
    assert min(floors) >= 77, (
        f"setuptools floor {min(floors)} predates PEP 639; the SPDX license "
        "string would fail on that backend"
    )


def _safe_is_file(path: Path) -> bool:
    """`is_file()` raises on entries Windows cannot stat.

    A WSL-created symlink left in a scratch directory makes `is_file()` raise
    WinError 1920 rather than return False. The leak scanner hit this first; the
    same guard is needed here because this test also walks the whole tree.
    """
    try:
        return path.is_file()
    except OSError:
        return False


def test_no_proprietary_string_anywhere() -> None:
    """A stale 'Proprietary' label contradicts an open-source repository.

    This file itself is excluded: it necessarily contains the word in order to
    search for it. Dot-directories (tooling scratch) are excluded by convention.
    """
    this_file = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    offenders: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not _safe_is_file(path) or path.suffix in {".pyc", ".pyo", ".safetensors", ".log"}:
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(p in {"runs", ".git", "__pycache__"} for p in rel.parts):
            continue
        if any(p.startswith(".") for p in rel.parts[:-1]):
            continue
        if rel.as_posix() == this_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if re.search(r"proprietary", line, re.I):
                offenders.append(f"{rel.as_posix()}:{lineno}")
    assert offenders == [], (
        "these files still claim a proprietary license:\n  " + "\n  ".join(offenders)
    )


def test_readme_points_at_a_real_license() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme[readme.rfind("## 许可证"):]
    assert "Apache-2.0" in section, "README should name the license"
    assert "（若有）" not in section, (
        "README hedges that the license may not exist, but LICENSE is present"
    )
    assert (REPO_ROOT / "LICENSE").is_file()


def _canonical_apache_text() -> str | None:
    """Find a dependency's copy of the Apache-2.0 text, if one is installed.

    Located through the installed distribution rather than a hardcoded venv path:
    the path is machine-specific, and hardcoding it would both leak this host's
    layout into a public repo and break for anyone else.
    """
    try:
        import importlib.metadata as md
    except ImportError:  # pragma: no cover
        return None
    for package in ("compressed-tensors", "llmcompressor", "safetensors", "huggingface-hub"):
        try:
            dist = md.distribution(package)
        except md.PackageNotFoundError:
            continue
        root = Path(dist.locate_file(""))
        if not root.is_dir():
            continue
        for candidate in (
            root / f"{package.replace('-', '_')}-{dist.version}.dist-info/licenses/LICENSE",
        ):
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="replace")
                idx = text.find("Apache License")
                if idx >= 0:
                    return text[idx:]
    return None


def test_license_matches_the_canonical_apache_text_except_the_holder() -> None:
    """Every line except the copyright notice must be byte-exact.

    Several installed packages ship identical Apache text; comparing against one
    proves ours was copied rather than retyped. The single permitted difference is
    the appendix copyright line, which this project is *required* to fill in.
    Skips when no such dependency is installed (e.g. a clone without the
    execution-layer stack).
    """
    canonical = _canonical_apache_text()
    if canonical is None:
        pytest.skip("no installed dependency ships the Apache-2.0 text")
    installed = LICENSE.read_text(encoding="utf-8")

    # Normalise BOTH sides by blanking the copyright line, then require equality.
    pattern = re.compile(r"(?m)^(\s*)Copyright\s+(?:\[yyyy\]\s+\[name of copyright owner\]|\d{4}\s+\S.*)$")

    def blank(text: str) -> str:
        return pattern.sub(lambda m: f"{m.group(1)}Copyright <HOLDER>", text)

    assert blank(installed) == blank(canonical), (
        "LICENSE differs from the canonical Apache-2.0 text somewhere other than "
        "the copyright holder line; the legal text must be byte-exact"
    )

    # And the only difference really is that one line.
    installed_lines = installed.splitlines()
    canonical_lines = canonical.splitlines()
    assert len(installed_lines) == len(canonical_lines), (
        "LICENSE has a different line count than the canonical text"
    )
    differing = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(installed_lines, canonical_lines), 1)
        if a != b
    ]
    assert len(differing) == 1, (
        f"expected exactly one differing line (the copyright notice), got {len(differing)}: "
        f"{[d[0] for d in differing]}"
    )
    line_no, ours, theirs = differing[0]
    assert "Copyright" in ours and "Copyright" in theirs, (
        f"the differing line {line_no} is not the copyright notice: {ours!r}"
    )