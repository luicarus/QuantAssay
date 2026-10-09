"""Atomic experiment evidence I/O and file checksums."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from quantassay.contracts import canonical_json


class StoreError(RuntimeError):
    """Raised for invalid run directories or broken state."""


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write via a temp file in the same directory, then atomically replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Deterministic JSON write, so checksums are stable."""
    return atomic_write_text(path, canonical_json(payload) + "\n")


def read_json(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.is_file():
        raise StoreError(f"expected file not found: {file_path}")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StoreError(f"corrupt JSON in {file_path}: {exc}") from exc


def file_sha256(path: str | Path) -> str:
    """Checksum used to confirm a cached artifact is the one we measured."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
